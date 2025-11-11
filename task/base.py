import random
from typing import Optional, Tuple

import numpy as np
import torch
from scipy.stats import truncnorm
from torch.utils.data import Dataset


def sample_truncnorm(size, mu=0, sigma=0.4, lower=-1, upper=1):
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma).rvs(size)


class TaskDataset(Dataset):
    """
    Abstract dataset class for generating embedding sequences for matching tasks.
    Each task is a sequence of trials where one discrete feature (e.g. category, identity, etc.)
    may repeat, indicating a match, and others vary.

    Attributes:
        dataset_size: Number of tasks in the dataset.
        task_len: Number of items per task.
        category_size, identity_size, position_size: Discrete feature vocab sizes.
        view_size, size_dim: Continuous feature dimensions.
        feature: The feature used to define "match" conditions.
    """

    def __init__(
            self,
            dataset_size: int,
            task_len: int = 6,
            category_size: int = 8,
            identity_size: int = 8,
            position_size: int = 9,
            view_size: int = 3,
            size_dim: int = 1,
            feature: str = 'category',
    ):
        assert position_size in [4, 9], 'position size should be 2x2 or 3x3 grid'
        self.feature = feature
        self.task_index = None
        self.task_len = task_len
        self.dataset_size = dataset_size

        # Discrete feature vocab sizes
        self.category_size = category_size
        self.identity_size = identity_size
        self.position_size = position_size
        # Continuous feature dimensions
        self.view_size = view_size
        self.size_dim = size_dim

        # Total embedding size (discrete + continuous parts)
        self.embedding_size = (
                category_size +
                category_size * identity_size +
                position_size +
                view_size +
                size_dim
        )

        self.cont_size = (dataset_size, task_len, view_size + size_dim)
        self.one_hot_size = (dataset_size, task_len, self.embedding_size - (view_size + size_dim))

        # Optionally apply noise to discrete embeddings
        one_hot = torch.zeros(self.one_hot_size)
        self.dataset = torch.cat([one_hot, torch.zeros(self.cont_size)], dim=-1)

        # Default action is "no match" (2)
        self.actions = torch.full((dataset_size, task_len), 2, dtype=torch.long)  # 2 = no action
        # self.reset()

    def available_categories(self):
        """Return available category ids (1-based)."""
        return list(range(1, self.category_size + 1))

    def available_identities(self, category: Optional[int] = None):
        """
        Return available identity ids (1-based).
        If category is provided, return identities available for that category.
        """
        return list(range(1, self.identity_size + 1))

    def available_positions(self, category: Optional[int] = None, identity: Optional[int] = None):
        """
        Always [1, ..., position size]
        """
        return list(range(1, self.position_size + 1))

    @staticmethod
    def _sample_from(choices, exclude: Optional[int] = None) -> int:
        """Return random element from `choices` excluding `exclude` (if present)."""
        if exclude is None:
            return random.choice(choices)
        filtered = set(choices)
        filtered.discard(exclude)
        if not filtered:
            raise RuntimeError(f"No alternative choice available (excluded={exclude})")
        return random.choice(list(filtered))

    def _set_random(self, data: torch.Tensor, category=None, identity=None, position=None, view=None, size=None):
        """
        Populate a single embedding with a random valid combination of features.
        Any feature left as None will be randomly chosen.
        """
        if category is None:
            cats = self.available_categories()
            category = int(self._sample_from(cats))

        if identity is None:
            ids = self.available_identities(category)
            identity = int(self._sample_from(ids))

        if position is None:
            poses = self.available_positions(category, identity)
            position = int(self._sample_from(poses))

        self._encode_discrete(data, category, identity, position)

        # sampling to match data distribution
        if view is None:
            if random.random() < 1 / 3:
                view = tuple(np.float32(np.random.uniform(-0.1, 0.1, 3)))
            else:
                view = tuple(sample_truncnorm(self.view_size))
        if size is None:
            if random.random() < 1 / 3:
                size = np.float32(np.random.uniform(-0.1, 0.1, 1))[0]
            else:
                size = sample_truncnorm(1)[0]

        self._encode_continuous(data, view, size)
        return category, identity, position, view, size

    def _get_new(
            self,
            prev: int,
            category: Optional[int] = None,
            identity: Optional[int] = None
    ) -> int:
        """
        Return a new available value (1-based) for the current feature, excluding `prev`.
        `category` is used when sampling identities (to pick identities available in that category).
        """
        if self.feature == "category":
            choices = self.available_categories()
            return int(self._sample_from(choices, exclude=prev))

        elif self.feature == "identity":
            # sample among identities available for the given category.
            choices = self.available_identities(category)
            return int(self._sample_from(choices, exclude=prev))
        else:  # position
            choices = self.available_positions(category, identity)
            return int(self._sample_from(choices, exclude=prev))

    def _set_data(
            self,
            data: torch.Tensor,
            category: int, identity: int, position: int,
            view=None, size=None,
    ):
        """
       Write one trial into the task sequence.
       With 50% chance, repeat a feature value (action = 1), else sample a new one (action = 0).
       """
        action = 0
        if random.random() < 0.5:  # match
            action = 1
            if self.feature == 'category':
                category, identity, position, view, size = self._set_random(
                    data, category=category
                )
            elif self.feature == 'identity':
                category, identity, position, view, size = self._set_random(
                    data, category=category, identity=identity
                )
            elif self.feature == 'position':
                category, identity, position, view, size = self._set_random(
                    data, position=position
                )
        else:  # no match, sample new feature
            if self.feature == 'category':
                new_category = self._get_new(category, self.category_size)
                category, identity, position, view, size = self._set_random(
                    data, category=new_category
                )
            elif self.feature == 'identity':
                new_category = random.choice(self.available_categories())
                if new_category == category:
                    new_identity = self._get_new(
                        identity, category=new_category
                    )
                else:
                    new_identity = self._sample_from(self.available_identities(category=new_category))
                category, identity, position, view, size = self._set_random(
                    data, category=new_category, identity=new_identity
                )
            elif self.feature == 'position':
                rand_category = random.choice(self.available_categories())
                rand_identity = self._sample_from(
                    self.available_identities(category=rand_category)
                )
                new_position = self._get_new(
                    position, category=rand_category, identity=rand_identity,
                )
                category, identity, position, view, size = self._set_random(
                    data,
                    category=rand_category,
                    identity=rand_identity,
                    position=new_position
                )
        return action, category, identity, position, view, size

    def _encode_discrete(self, data: torch.Tensor, category: int, identity: int, position: int):
        """
        Encode category, identity, and position into one-hot vectors in `data`.
        """
        # category one-hot
        data[:self.category_size].zero_()
        data[category - 1] = 1

        # identity block
        identity_start = self.category_size
        identity_end = identity_start + self.category_size * self.identity_size
        data[identity_start:identity_end].zero_()
        identity_offset = identity_start + (category - 1) * self.identity_size + identity - 1
        data[identity_offset] = 1

        position_start = identity_end
        position_end = position_start + self.position_size
        data[position_start:position_end].zero_()
        data[position_start + position - 1] = 1
        return

    def _encode_continuous(self, data: torch.Tensor, view: Tuple[float, ...], size: float):
        """
        Encode view and size features into the final segment of the embedding vector.
        """
        start = self.embedding_size - (self.view_size + self.size_dim)
        if self.view_size > 0:
            data[start:start + self.view_size] = torch.tensor(view, dtype=data.dtype)
        if self.size_dim == 1:
            data[start + self.view_size] = size

    def decode_embedding(self, embedding: torch.Tensor):
        """
        Decode an embedding vector back into feature values.
        """
        if embedding.ndim != 1:
            raise ValueError(f"Expected 1D embedding tensor, got shape {embedding.shape} instead")
        category_end = self.category_size
        identity_end = category_end + self.category_size * self.identity_size
        position_end = identity_end + self.position_size
        cont_start = position_end

        if embedding[:category_end].sum().item() > 1:
            raise RuntimeError("Invalid category one-hot (more than one hot)")
        category = torch.argmax(embedding[:category_end]).item()
        cat_1b = category + 1

        # Extract full identity slice first, then index
        full_identity = embedding[category_end:identity_end]
        identity_offset = category * self.identity_size
        identity_slice = full_identity[identity_offset:identity_offset + self.identity_size]
        if identity_slice.sum().item() > 1:
            raise RuntimeError("Invalid identity one-hot (more than one hot)")
        identity = torch.argmax(identity_slice).item()
        id_1b = identity + 1

        if embedding[identity_end:position_end].sum().item() > 1:
            raise RuntimeError("Invalid position one-hot (more than one hot)")
        position = torch.argmax(embedding[identity_end:position_end]).item()
        pos_1b = position + 1

        view = tuple(embedding[cont_start:cont_start + self.view_size].tolist())
        size = embedding[cont_start + self.view_size].item()
        return cat_1b, id_1b, pos_1b, view, size

    def batch_decode(self, emb: torch.Tensor):
        N, D = emb.shape
        category_end = self.category_size
        identity_end = category_end + self.category_size * self.identity_size
        position_end = identity_end + self.position_size
        cont_start = position_end

        cats = emb[:, :category_end].argmax(dim=1)
        full_identity = emb[:, category_end:identity_end].reshape(N, self.category_size, self.identity_size)
        ids = full_identity[torch.arange(N, device=emb.device), cats].argmax(dim=1)  # (N,)
        pos = emb[:, identity_end:position_end].argmax(dim=1)  # (N,)
        views = emb[:, cont_start: cont_start + self.view_size]  # (N, view_size)
        sizes = emb[:, cont_start + self.view_size]  # (N,)

        cat_1b = (cats + 1)
        id_1b = (ids + 1)
        pos_1b = (pos + 1)
        return cat_1b, id_1b, pos_1b, views, sizes

    def make_stub(self, N):
        emb = torch.zeros(N, self.embedding_size)

        # Fill with random one-hot blocks for categories, identities, positions
        # Category block
        for i in range(N):
            emb[i, torch.randint(0, self.category_size, (1,)).item()] = 1

        # Identity block
        identity_start = self.category_size
        identity_end = identity_start + self.category_size * self.identity_size
        for i in range(N):
            idx = torch.randint(0, self.category_size * self.identity_size, (1,)).item()
            emb[i, identity_start + idx] = 1

        pos_start = identity_end
        pos_end = pos_start + self.position_size
        for i in range(N):
            emb[i, pos_start + torch.randint(0, self.position_size, (1,)).item()] = 1

        view_start = pos_end
        view_end = view_start + self.view_size
        emb[:, view_start:view_end] = torch.rand(N, self.view_size)

        # Sizes (continuous scalar)
        emb[:, view_end] = torch.rand(N)
        return emb

    @staticmethod
    def get_stimuli_pair(trial_info):
        raise NotImplementedError

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        data = self.dataset[idx]
        action = self.actions[idx]
        return data, action, torch.tensor(self.task_index)

    def __iter__(self):
        for i in range(self.dataset_size):
            yield self.dataset[i], self.actions[i], torch.tensor(self.task_index)

    def reset(self):
        self.dataset = torch.zeros((self.dataset_size, self.task_len, self.embedding_size))
        self.actions = torch.full((self.dataset_size, self.task_len), 2, dtype=torch.long)


class HvMTaskDataset(TaskDataset):
    """
    TaskDataset that draws from real HvM images/metadata.

    - `_set_random`: samples an HvM row that matches any specified discrete features.
    - `_get_new`: selects a new discrete feature value that is valid in the HvM cache.
    - `_set_data`: inherited, builds match/non-match sequences using `_set_random` and `_get_new`.
    """

    def __init__(
            self,
            hvm_loader,
            dataset_size: int,
            task_len: int = 6,
            feature: str = "category",
            grid_size: int = 3,
    ):
        # Prepare the loader cache (positions, continuous features, optional images)
        self.cache = hvm_loader._task_cache
        self.hvm_loader = hvm_loader
        self.feature = feature

        super().__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=self.cache.category_size,
            identity_size=self.cache.identity_size,
            position_size=self.cache.position_size,
            view_size=self.cache.view_size,
            size_dim=self.cache.size_dim,
            feature=feature,
        )

        # Override embedding size
        self.embedding_size = self.cache.embedding_size
        # Precompute arrays for efficient masking
        self._cats = self.cache.cat_1b
        self._ids = self.cache.id_1b
        self._poses = self.cache.pos_1b
        self._views = self.cache.views
        self._sizes = self.cache.sizes
        self._vars = self.cache.vars
        self._valid_triples = self.cache.valid_triples
        self._row_indices = np.arange(len(self._cats))

        self.categories = [
            c for c in range(1, self.category_size + 1)
        ]
        self.ids = {c: list() for c in range(1, self.category_size + 1)}
        self.pos = self.cache.catid_to_positions

        self.global_pos = set(range(1, self.position_size + 1))
        for c in self.categories:
            for i in range(1, self.identity_size + 1):
                if np.where((self._ids == i) & (self._cats == c))[0].size > 0:
                    self.ids[c].append(i)
                key = (c, i)
                if key in self.pos:
                    self.global_pos &= set(self.pos[key])

    def available_categories(self):
        return self.categories

    def available_identities(self, category: Optional[int] = None):
        if category is None:
            return np.unique(self._ids)
        return self.ids[category]

    def available_positions(self, category=None, identity=None):
        if category is None or identity is None:
            print('sampling random pos')
            return self.global_pos
        return self.pos[(category, identity)]

    def _set_random(
            self,
            data: torch.Tensor,
            category: Optional[int] = None,
            identity: Optional[int] = None,
            position: Optional[int] = None,
            view=None,
            size=None
    ) -> Tuple[int, int, int, tuple, float]:
        """
        Samples a single HvM row matching any fixed discrete features.
        Leaves unspecified ones to vary.
        """
        # Start with all rows
        mask = np.ones_like(self._row_indices, dtype=bool)

        if category is not None:
            mask &= (self._cats == category)
        if identity is not None:
            mask &= (self._ids == identity)
        if position is not None:
            mask &= (self._poses == position)

        candidates = self._row_indices[mask]
        if len(candidates) == 0:
            raise RuntimeError(
                f"No HvM images match category={category}, identity={identity}, position={position}"
            )

        cand_vars = self._vars[candidates]
        weights = np.ones_like(cand_vars, dtype=np.float32)
        # weights[cand_vars == 6] = 0.5  # pick var6 2x less often

        probs = weights / weights.sum()
        idx = int(np.random.choice(candidates, p=probs))

        # Extract features from that row
        category, identity, position = int(self._cats[idx]), int(self._ids[idx]), int(self._poses[idx])
        view = tuple(self._views[idx]) if self._views is not None else ()
        size = float(self._sizes[idx]) if self._sizes is not None else 0.0
        assert (category, identity, position) in self._valid_triples
        # Encode into data
        self._encode_discrete(data, category, identity, position)
        self._encode_continuous(data, view, size)

        return category, identity, position, view, size
