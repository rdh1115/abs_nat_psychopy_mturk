import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence, Union, List

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import v2

from task.base import HvMTaskDataset


def make_read_fn(img_size=224, dataset_stats=None, transform=None):
    if dataset_stats is None:
        dataset_stats = ([0, 0, 0],
                         [1, 1, 1])
        # dataset_stats = ([0.485, 0.456, 0.406],
        #                  [0.229, 0.224, 0.225])
    print(f'normalizing with stats: {dataset_stats}')
    mean, std = dataset_stats
    default_resize = v2.Resize((img_size, img_size), antialias=True)
    default_normalize = v2.Normalize(mean, std)

    def _read(p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError(f"cv2 failed to load image at: {p}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).permute(2, 0, 1).float().div(255.0)

        if transform is not None:
            # use the provided transform pipeline instead of default resize+normalize
            im = transform(im)
        else:
            im = default_resize(im)
            im = default_normalize(im)
        return im.contiguous()

    return _read


def save_train_val_split(
        df,
        out_dir=None,
        val_ratio=0.2
):
    """
    Splits the HvM dataframe into train/val sets by objects within each category.
    Ensures both splits contain all categories but disjoint objects.

    Args:
        df (pd.DataFrame): DataFrame with columns including 'category' and 'object'.
        out_dir (Path | str | None): Directory to save CSVs. If None, saves into the dataset's root.
        val_ratio (float): Fraction of objects per category for validation.
    """
    df = df.copy()
    df["split"] = None

    for cat, cat_df in df.groupby("category_id"):
        objects = cat_df["obj_id"].unique()
        np.random.shuffle(objects)

        n_val = max(1, int(len(objects) * val_ratio))
        val_objs = set(objects[:n_val])

        df.loc[(df["category_id"] == cat) & (df["obj_id"].isin(val_objs)), "split"] = "val"
        df.loc[(df["category_id"] == cat) & (~df["obj_id"].isin(val_objs)), "split"] = "train"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    return train_df, val_df


@dataclass
class HvMTaskCache:
    # sizes
    grid_size: int
    category_size: int
    identity_size: int
    position_size: int

    view_size: int
    size_dim: int
    embedding_size: int

    # dataframe-derived
    df_idx: np.ndarray
    cat_1b: np.ndarray
    id_1b: np.ndarray
    pos_1b: np.ndarray
    views: np.ndarray
    sizes: np.ndarray
    vars: np.ndarray
    filenames: np.ndarray

    # maps
    valid_triples: set
    triple_to_indices: Dict[Tuple[int, int, int], np.ndarray]
    cat_to_ids: Dict[int, np.ndarray]
    catid_to_positions: Dict[Tuple[int, int], np.ndarray]

    # images cache
    images: Optional[torch.Tensor] = None  # [N, C, H, W]
    filename_to_idx: Optional[Dict[str, int]] = None  # path -> row index


class HvMMetaData:
    """
    build df and images for all stimuli in the stim_dir
    shared across datasets
    """

    def __init__(
            self,
            root_dir: Union[str, Path],
            normalize: str = "minmax",
            grid_size=3,
    ):
        self.root_dir = Path(root_dir)
        self.grid_size = grid_size
        self.meta_dir = self.root_dir / "python"
        self.stim_dir = self.root_dir / "HvM_with_discfade"
        self.normalize = normalize
        self.filtered = False

        self.df: Optional[pd.DataFrame] = None
        self.images = None
        self.filename_to_idx = None
        self.load_metadata()
        self.filter_files()

    def load_metadata(self):
        """
         Load and optionally normalize metadata from selected variation .h5 files.
         Builds mappings and label-to-path dictionary.
         """
        h5_files, h5_vars = self.validate_structure()

        all_dfs = dict()
        for var, fp in zip(h5_vars, h5_files):
            df = convert_h5_to_df(fp)

            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)

            if 'filename' in df.columns:
                df['filename'] = df['filename'].map(partial(localize_h5_filename, fp))

            df['var'] = var
            all_dfs[var] = df

        self.df = pd.concat(all_dfs.values(), ignore_index=True)

        if self.normalize is not None:
            float_cols = ['rxy', 'rxz', 'ryz', 's']
            for col in float_cols:
                if self.normalize == 'minmax':  # Normalize all float columns to [-1, 1]
                    min_val = self.df[col].min()
                    max_val = self.df[col].max()
                    if max_val > min_val:  # avoid divide by zero
                        self.df[col] = 2 * (self.df[col] - min_val) / (max_val - min_val) - 1
                elif self.normalize == 'z-score':
                    mean_val = self.df[col].mean()
                    std_val = self.df[col].std()
                    if std_val > 0:  # avoid divide by zero
                        self.df[col] = (self.df[col] - mean_val) / std_val
                else:
                    raise ValueError('normalize with minmax or z-score')

        if 'category' not in self.df or 'obj' not in self.df:
            raise ValueError("Metadata must contain 'category' and 'obj' columns")

        self._build_mappings()
        # 1-based IDs
        self.df["cat_1b"] = self.df["category_id"].astype(int) + 1
        self.df["id_1b"] = self.df["obj_id"].astype(int) + 1
        self.df["pos_1b"] = self._build_discrete_positions(self.df, "ty", "tz", self.grid_size)
        return self.df

    def validate_structure(self):
        h5_files = sorted(self.meta_dir.glob('hvmmeta_var*.h5'))
        h5_vars = [int(f.stem.split('hvmmeta_var')[1]) for f in h5_files]

        variation_folders = sorted([
            f for f in self.stim_dir.iterdir()
            if f.is_dir() and f.name.startswith('Variation')
        ])
        folder_vars = [int(f.name.replace('Variation', '')[:2]) for f in variation_folders]

        if set(h5_vars) != set(folder_vars):
            raise ValueError(f"Mismatch between h5 files {h5_vars} and variation folders {folder_vars}")

        return h5_files, h5_vars

    def _build_mappings(self):
        """
        Build mappings from string category and obj labels to integer IDs.
        Each obj_id is unique *within* its category (i.e., 0, 1, 2,... per category).
        Adds new columns `category_id` and `obj_id` to self.df.
        """
        # Map categories to IDs
        unique_categories = sorted(self.df['category'].unique())
        self.category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
        self.df['category_id'] = self.df['category'].map(self.category_to_id)

        self.obj_to_id = {}
        for cat, group in self.df.groupby('category'):
            objs_in_cat = sorted(group['obj'].unique())
            obj_map = {obj: i for i, obj in enumerate(objs_in_cat)}
            # Store mapping per category
            self.obj_to_id[cat] = obj_map

        self.df['obj_id'] = self.df.apply(
            lambda row: self.obj_to_id[row['category']][row['obj']], axis=1
        )
        return

    def check_file_existence(self):
        self.df['file_exists'] = self.df['filename'].apply(lambda p: Path(p).exists())

    def filter_files(self):
        """
        Filters self.df in-place to only include rows where files exist.
        """
        if self.df is None:
            self.load_metadata()
        else:
            if self.filtered:
                return self.df
        # Ensure file_exists column is present
        if 'file_exists' not in self.df.columns:
            self.check_file_existence()

        # Filter
        no_local_files = self.df[~self.df['file_exists']]
        print(f'removed {len(no_local_files)} rows from the original df')
        self.df = self.df[self.df['file_exists']].reset_index(drop=True)
        print(f'found {len(self.df)} local images after filtering')
        self.filtered = True
        return self.df

    def build_image_cache(
            self,
            img_read_fn: callable,
            dtype: torch.dtype = torch.float32,
    ):
        imgs = []
        filenames = self.df["filename"].astype(str).to_numpy()
        for i, p in enumerate(filenames):
            im = img_read_fn(p)
            imgs.append(im)
        img_tensor = torch.stack(imgs, dim=0).to(dtype).contiguous()  # [N, H, W, C]
        filename_to_idx = {fn: i for i, fn in enumerate(filenames)}
        self.images = img_tensor.share_memory_()
        self.filename_to_idx = filename_to_idx
        return img_tensor, filename_to_idx

    @staticmethod
    def _build_discrete_positions(df, ty_col: str, tz_col: str, grid_size: int) -> np.ndarray:
        tx = df[ty_col].to_numpy()  # ty_col actually stores horizontal position
        ty = df[tz_col].to_numpy()  # tz_col actually stores vertical position

        tx_edges = np.linspace(tx.min(), tx.max(), grid_size + 1)
        ty_edges = np.linspace(ty.min(), ty.max(), grid_size + 1)

        tx_bin = np.clip(np.digitize(tx, tx_edges[:-1], right=False), 1, grid_size)
        ty_bin = np.clip(np.digitize(ty, ty_edges[:-1], right=False), 1, grid_size)

        tx_bin_flipped = grid_size + 1 - tx_bin
        ty_bin_flipped = grid_size + 1 - ty_bin  # this doesn't affect tasks, just for human visualization
        pos = (ty_bin_flipped - 1) * grid_size + tx_bin_flipped
        return pos.astype(int)


class HvMImageLoader:
    """
    used for reading images and initializing HvmTasks
    """

    def __init__(
            self,
            root_dir=None,
            metadata: Optional[HvMMetaData] = None,
            mode: str = 'training',
            preload_images=True,
            transform=None,
            img_size=224,
            df: pd.DataFrame = None,
            pretrained_stats=None
    ):
        """
        Args:
            root_dir (str or Path, optional):
                Root directory containing 'python/' and 'HvM_with_discfade/'.
                Defaults to project_root / 'data' / 'original'.
        """
        if root_dir is None:
            # Determine project root by going up from current file location
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[1]
            root_dir = project_root / 'data' / 'original'

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.preload_images = preload_images
        self.img_size = img_size
        self.pretrained_stats = pretrained_stats
        self.transform = transform
        self.img_read_fn = make_read_fn(
            self.img_size,
            self.pretrained_stats,
            self.transform
        )

        self.metadata = metadata if metadata is not None else HvMMetaData(root_dir=self.root_dir)
        if self.preload_images:
            if self.metadata.images is None:
                self.metadata.build_image_cache(self.img_read_fn)
        self.meta_dir = self.metadata.meta_dir
        self.stim_dir = self.metadata.stim_dir

        self.df = df if df is not None else self.metadata.df

        if self.mode == 'debug':
            self.accessed_paths = dict.fromkeys(self.df['filename'].astype(str).unique(), 0)

    @classmethod
    def from_train_val(
            cls,
            root_dir: Union[str, Path] = None,
            out_dir: Union[str, Path] = None,
            val_ratio: float = 1 / 4,
            train_vars=(0, 3, 6),
            val_vars=(0, 3, 6),
            preload_images: bool = True,
            grid_size: int = 3,
            img_size: int = 224,
            pretrained_stats=None,
            transform=None,
    ) -> Tuple["HvMImageLoader", "HvMImageLoader"]:
        """
        Automatically create train and val loaders.
        - Looks for train.csv and val.csv in stim_dir.
        - If missing, splits metadata.df and saves CSVs.
        """
        if root_dir is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parents[1]
            root_dir = project_root / 'data' / 'original'
        meta = HvMMetaData(root_dir, grid_size=grid_size)
        if preload_images:
            img_read_fn = make_read_fn(img_size, pretrained_stats, transform)
            meta.build_image_cache(img_read_fn=img_read_fn)

        stim_dir = meta.stim_dir
        train_csv = stim_dir / "train.csv"
        val_csv = stim_dir / "val.csv"

        # build CSVs if missing
        if not train_csv.exists() or not val_csv.exists():
            train_df, val_df = save_train_val_split(
                meta.df,
                out_dir=stim_dir if out_dir is None else out_dir,
                val_ratio=val_ratio
            )
            print(f"Saved train.csv ({len(train_df)}) and val.csv ({len(val_df)})")
        else:
            print(f'Found train split csv at {train_csv}')
            print(f'Found validation split csv at {val_csv}')
            train_df = pd.read_csv(train_csv)
            val_df = pd.read_csv(val_csv)
            train_df['filename'] = train_df['filename'].map(partial(localize_csv_filename, stim_dir))
            val_df['filename'] = val_df['filename'].map(partial(localize_csv_filename, stim_dir))

        # instantiate loaders
        train_loader = cls(
            root_dir=root_dir,
            metadata=meta,
            preload_images=preload_images,
            df=train_df,
            img_size=img_size,
            pretrained_stats=pretrained_stats,
            transform=transform
        )
        val_loader = cls(
            root_dir=root_dir,
            metadata=meta,
            preload_images=preload_images,
            df=val_df,
            img_size=img_size,
            pretrained_stats=pretrained_stats,
            transform=transform
        )

        train_loader.prepare_for_tasks(
            grid_size,
            filter_vars=train_vars
        )
        val_loader.prepare_for_tasks(
            grid_size,
            filter_vars=val_vars
        )
        return train_loader, val_loader

    def clear_caches(self):
        self._loc_cache = dict()
        self._view_cache = dict()
        self._size_cache = dict()

    def load_image(self, path: str) -> torch.Tensor:
        """Load image either from memory cache or disk (cv2)."""

        path = str(path)
        if hasattr(self, "_task_cache") and self._task_cache.images is not None:
            idx = self._task_cache.filename_to_idx[path]
            if self.mode == 'debug':
                self.accessed_paths[path] += 1
            return self._task_cache.images[idx]
        else:
            # fallback: original behavior
            if self.mode == 'debug':
                self.accessed_paths[path] += 1

            if not Path(path).exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = self.read_img(path)
            return image

    def batch_load_image(self, paths):
        """
        find the image tensors from the provided paths
        """
        if hasattr(self, "_task_cache") and self._task_cache.images is not None:
            paths = [str(p) for p in paths]
            idxs = [self._task_cache.filename_to_idx[p] for p in paths]
            images = self._task_cache.images[idxs]
        else:
            images = list()
            for p in paths:
                images.append(self.load_image(p))
            images = torch.stack(images)
        return images

    def read_img(self, p):
        im = self.img_read_fn(p)
        return im.contiguous()

    def prepare_for_tasks(
            self,
            grid_size: int = 3,
            view_cols: Sequence[str] = ("rxy", "rxz", "ryz"),
            size_col: Optional[str] = "s",
            filter_vars=(0, 3, 6),
    ):
        """
        Build (and cache) everything HvMTaskDataset needs.
        If `preload_images=True`,
        we read all images once with cv2.imread and keep them in RAM.
        """
        if hasattr(self, "_task_cache") and self._task_cache is not None:
            tc = self._task_cache
            return tc  # already prepared

        preload_images = self.preload_images
        task_df = self.df.copy()
        task_df = task_df.loc[task_df['var'].isin(filter_vars)]

        if not {"category_id", "obj_id", "ty", "tz"}.issubset(task_df.columns):
            raise ValueError("df must already have category_id, obj_id, ty, tz columns.")

        category_size = 8
        identity_size = 8
        position_size = grid_size * grid_size
        have_views = all(c in task_df.columns for c in view_cols)
        view_size = len(view_cols) if have_views else 0
        size_dim = 1 if (size_col is not None and size_col in task_df.columns) else 0

        embedding_size = (
                category_size +
                category_size * identity_size +
                position_size +
                view_size +
                size_dim
        )

        # maps
        triple_to_indices = defaultdict(list)
        cat_to_ids = defaultdict(set)
        catid_to_positions = defaultdict(set)
        row_indices = np.arange(len(task_df))

        for i, r in task_df.iterrows():
            c, ident, p = int(r["cat_1b"]), int(r["id_1b"]), int(r["pos_1b"])
            triple_to_indices[(c, ident, p)].append(i)
            cat_to_ids[c].add(ident)
            catid_to_positions[(c, ident)].add(p)

        valid_triples = set(triple_to_indices.keys())
        triple_to_indices = {k: np.asarray(v, dtype=np.int64) for k, v in triple_to_indices.items()}
        cat_to_ids = {c: np.asarray(sorted(list(v)), dtype=np.int64) for c, v in cat_to_ids.items()}
        catid_to_positions = {
            k: np.asarray(sorted(list(v)), dtype=np.int64) for k, v in catid_to_positions.items()
        }

        # continuous
        views = task_df.loc[:, view_cols].to_numpy(dtype=np.float32) if view_size > 0 else None
        sizes = task_df[size_col].to_numpy(dtype=np.float32) if size_dim == 1 else None
        filenames = task_df["filename"].astype(str).to_numpy()
        vars = task_df["var"].to_numpy(dtype=np.int8)

        images_np = None
        filename_to_idx = None
        if preload_images:
            print('Using preloaded np img arrays')

            images_np = self.metadata.images
            filename_to_idx = self.metadata.filename_to_idx
        else:
            if self.mode == 'debug':
                print('Getting np img arrays from io')
        self.df = task_df

        self._task_cache = HvMTaskCache(
            grid_size=grid_size,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            view_size=view_size,
            size_dim=size_dim,
            embedding_size=embedding_size,
            df_idx=row_indices,
            cat_1b=task_df["cat_1b"].to_numpy(dtype=np.int8),
            id_1b=task_df["id_1b"].to_numpy(dtype=np.int8),
            pos_1b=task_df["pos_1b"].to_numpy(dtype=np.int8),
            views=views,
            sizes=sizes,
            vars=vars,
            filenames=filenames,

            valid_triples=valid_triples,
            triple_to_indices=triple_to_indices,
            cat_to_ids=cat_to_ids,
            catid_to_positions=catid_to_positions,

            images=images_np,
            filename_to_idx=filename_to_idx
        )
        return

    def get_path_coverage(self):
        """
        Visualize and print accessed vs unaccessed file distributions.
        """
        if not self.mode == 'debug':
            return
        accessed = {k for k, v in self.accessed_paths.items() if v > 0}
        unaccessed = {k for k, v in self.accessed_paths.items() if v == 0}

        df_accessed = self.df[self.df['filename'].astype(str).isin(accessed)]
        df_unaccessed = self.df[self.df['filename'].astype(str).isin(unaccessed)]
        # plot_continuous_attribute_distributions(df_accessed)
        # plot_continuous_attribute_distributions(df_unaccessed)
        coverage = len(accessed) / len(self.df['filename'].unique())
        print(f"Image coverage: {coverage:.2%}")
        return coverage

    def clear_paths(self):
        if hasattr(self, "accessed_paths"):
            self.accessed_paths = {k: 0 for k in self.accessed_paths.keys()}

    def get_all_loc_for_object(self, category, obj, sample_vars: list[int] = None, normalize=True):
        """
        Returns a list of tuples [(ty, tz, path), ...] for all images matching discrete keys.
        """
        if not hasattr(self, '_loc_cache'):
            self._loc_cache = {}
        if sample_vars is None:
            sample_vars = [0, 3, 6]
        cache_key = (category, obj, tuple(sample_vars), normalize)
        if cache_key in self._loc_cache:
            return self._loc_cache[cache_key]

        df = self.df
        rows = df[(df['category_id'] == category) & (df['obj_id'] == obj)].copy()

        if rows.empty:
            return []
        if normalize:
            for key in ['ty', 'tz']:
                col_min, col_max = df[key].min(), df[key].max()
                if col_max > col_min:
                    rows[key] = 2 * (rows[key] - col_min) / (col_max - col_min) - 1

        result = list(zip(rows['ty'], rows['tz'], rows['filename'].astype(str)))
        self._loc_cache[cache_key] = result
        return result

    def get_all_view_angles_for_object(self, category, obj, sample_vars: list[int] = None, normalize=True):
        """
        Returns a list of tuples [(rxy, rxz, ryz, path), ...] for all images matching discrete keys.
        """
        if not hasattr(self, '_view_cache'):
            self._view_cache = {}
        if sample_vars is None:
            sample_vars = [0, 3, 6]
        cache_key = (category, obj, tuple(sample_vars), normalize)
        if cache_key in self._view_cache:
            return self._view_cache[cache_key]

        df = self.df
        rows = df[(df['category_id'] == category) & (df['obj_id'] == obj)].copy()
        if rows.empty:
            return []

        if normalize:
            for key in ['rxy', 'rxz', 'ryz']:
                col_min, col_max = df[key].min(), df[key].max()
                if col_max > col_min:
                    rows[key] = 2 * (rows[key] - col_min) / (col_max - col_min) - 1

        result = list(zip(rows['rxy'], rows['rxz'], rows['ryz'], rows['filename'].astype(str)))
        self._view_cache[cache_key] = result
        return result

    def get_all_s_for_object(self, category, obj, sample_vars: list[int] = None, normalize=True):
        """
        Returns a list of tuples [(s, path), ...] for all images matching discrete keys.
        """
        if not hasattr(self, '_size_cache'):
            self._size_cache = {}
        if sample_vars is None:
            sample_vars = [0, 3, 6]
        cache_key = (category, obj, tuple(sample_vars), normalize)
        if cache_key in self._size_cache:
            return self._size_cache[cache_key]

        df = self.df
        rows = df[(df['category_id'] == category) & (df['obj_id'] == obj)].copy()

        if rows.empty:
            return []
        if normalize:
            s_min, s_max = df['s'].min(), df['s'].max()
            if s_max > s_min:
                rows['s'] = 2 * (rows['s'] - s_min) / (s_max - s_min) - 1

        result = list(zip(rows['s'], rows['filename'].astype(str)))
        self._size_cache[cache_key] = result
        return result


class HvMImageMapper:
    """
    given embeddings, returns image tensors.
    """

    def __init__(self, task_dataset: HvMTaskDataset, noise_fn=None, img_size=224):
        self.task_dataset = task_dataset
        self.img_size = img_size
        self.loader = task_dataset.hvm_loader
        self.cache = task_dataset.cache
        self.noise_fn = noise_fn
        self.row_cache = dict()

    def find(self, decode_tuple):
        cat, obj, pos, view, size = decode_tuple
        cat, obj, pos, size = cat.item(), obj.item(), pos.item(), size.item()
        view = tuple(e.item() for e in view)
        decode_tuple = (cat, obj, pos, view, size)
        if decode_tuple in self.row_cache:
            return self.row_cache[decode_tuple]

        cat, obj, pos, view, size = decode_tuple

        mask = (
                (self.cache.cat_1b == int(cat)) &
                (self.cache.id_1b == int(obj)) &
                (self.cache.pos_1b == int(pos))
        )
        if self.cache.views is not None:
            mask &= np.all(self.cache.views == np.array(view, dtype=self.cache.views.dtype), axis=1)
        if self.cache.sizes is not None:
            mask &= (self.cache.sizes == float(size))
        idxs = np.where(mask)[0]

        if idxs.size == 0:
            raise RuntimeError(f"No image for category {cat}, id {obj}, pos {pos}, view {view}, size {size}")
        idx = int(idxs[0])
        self.row_cache[decode_tuple] = idx
        return int(idxs[0])

    def _decode_and_find(self, emb):
        decode_tuple = self.task_dataset.decode_embedding(emb)
        file = self.cache.filenames[self.find(decode_tuple)]
        return file, decode_tuple

    def _batch_decode_and_find(self, embs):
        decode_tuples = self.task_dataset.batch_decode(embs)
        files = list()
        for decode_tuple in zip(*decode_tuples):
            files.append(self.cache.filenames[self.find(decode_tuple)])
        return files, decode_tuples

    def _load_one(self, path: str) -> torch.Tensor:
        img = self.loader.load_image(path)
        if self.noise_fn:
            img = self.noise_fn(img)
        return img

    def _load_batch(self, paths):
        imgs = self.loader.batch_load_image(paths)
        if self.noise_fn:
            imgs = self.noise_fn(imgs)
        return imgs

    def map_sequence(self, emb_arr):
        """
        emb_arr:   array of shape (T, embed_dim)
        """
        N = emb_arr.shape[0]
        C, H, W = 3, self.img_size, self.img_size
        imgs_out = torch.zeros((N, C, H, W), dtype=torch.float32)
        task_info = {i: None for i in range(N)}

        zero_mask = torch.all(torch.isclose(
            emb_arr,
            torch.tensor(0.0, dtype=emb_arr.dtype)
        ), dim=1)
        nonzero_idx = torch.nonzero(~zero_mask).squeeze(1)
        assert not nonzero_idx.numel() == 0, 'this should not happen'

        subset = emb_arr[nonzero_idx]
        files, decode_tuples = self._batch_decode_and_find(subset)
        imgs = self._load_batch(files)
        per_sample_decodes = list(zip(*decode_tuples))

        for j, flat_pos in enumerate(nonzero_idx.tolist()):
            imgs_out[flat_pos] = imgs[j]
            task_info[flat_pos] = per_sample_decodes[j]
        return imgs_out, task_info

    def map_batch(self, emb_batch: np.ndarray):
        """
        emb_batch: array of shape (B, T, embed_dim)
        returns:   tensor of shape (B, T, 3, H, W)
        """
        B, T, D = emb_batch.shape
        flat = emb_batch.reshape(B * T, D)  # (B*T, D)
        imgs_flat, task_info_flat = self.map_sequence(flat)  # imgs_flat: (B*T, C, H, W)

        imgs = imgs_flat.view(B, T, *imgs_flat.shape[1:])
        task_info = {
            b: {
                t: task_info_flat[b * T + t]
                for t in range(T)
            }
            for b in range(B)
        }
        return imgs, task_info


def localize_h5_filename(h5_fp, val):
    local_base = h5_fp.parent.parent / 'HvM_with_discfade'
    local_fp = local_base / Path(*Path(val).parts[5:])
    return local_fp


def _subpath_after(p: Path, segment: str) -> Optional[Path]:
    """Return the subpath (as Path) after `segment`"""
    try:
        idx = p.parts.index(segment)
    except ValueError:
        return None
    return Path(*p.parts[idx + 1:])


def localize_csv_filename(
        stim_dir: Union[str, Path],
        val: Union[str, Path],
        segment: str = "HvM_with_discfade",
) -> Path:
    """
    Build a local path by taking the portion of `val` after `segment`
    and anchoring it under the occurrence of `segment` found in `h5_fp`.

    on_missing:
      - "raise"  : raise ValueError if `segment` is missing in either path (default).
      - "basename": if `val` lacks the segment, use val's filename as the relative part.
                   if `stim_dir` lacks the segment, use stim_dir.parent as the anchor.
      - "use_val" : if stim_dir lacks the segment, return Path(val) unchanged.
                   if val lacks the segment, return Path(val) unchanged.
    """
    stim_dir = Path(stim_dir)
    val_p = Path(val)
    dir_idx = stim_dir.parts.index(segment)
    stim_base = Path(*stim_dir.parts[: dir_idx + 1])

    relative = _subpath_after(val_p, segment)
    return stim_base / relative


def convert_h5_to_df(fp):
    with h5py.File(fp, 'r') as f:
        dfs = {k: pd.DataFrame(np.array(f[k]), columns=[k]) for k in f.keys()}
    return pd.concat(dfs.values(), axis=1)
