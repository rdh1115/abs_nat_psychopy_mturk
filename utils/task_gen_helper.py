import itertools
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pulp import (
    LpProblem,
    LpVariable,
    LpBinary,
    lpSum,
    LpStatus,
    LpMinimize,
)

from task.base import HvMTaskDataset
from task.multi_task import TASK_NAME_TASK_INDEX
from utils.helper import _subpath_after, TaskConfig
from utils.stim_io import HvMMetaData, HvMImageLoader, HvMImageMapper


def pick_objs(df, n_objs):
    catids = list()
    cats = sorted(df['cat_1b'].unique())

    obj_per_cat = n_objs // len(cats)
    remainder = n_objs % len(cats)

    # main balanced allocation
    for cat in cats:
        objs_in_cat = sorted(df.loc[df['cat_1b'] == cat, 'id_1b'].unique())
        chosen = random.sample(objs_in_cat, obj_per_cat)
        for obj in chosen:
            catids.append((cat, obj))

    # randomly choose remainder categories
    if remainder > 0:
        extra_cats = random.sample(cats, k=remainder)  # 👈 random instead of cats[:remainder]
        for cat in extra_cats:
            objs_in_cat = sorted(df.loc[df['cat_1b'] == cat, 'id_1b'].unique())
            already = {obj for (c, obj) in catids if c == cat}
            available = list(set(objs_in_cat) - already)
            if available:
                catids.append((cat, random.choice(available)))
    return catids


def pick_locations(df, catid_to_positions, chosen_objs, loc_c=5):
    rows = list()
    for catid in chosen_objs:
        cat, obj = catid
        positions = sorted(catid_to_positions[catid])

        # make sure there is a consistent location for task sampling
        chosen_pos = [loc_c]
        positions.remove(loc_c)
        chosen_pos.append(random.sample(positions, k=1)[0])
        for pos in chosen_pos:
            subset = df[
                (df["cat_1b"] == cat) &
                (df["id_1b"] == obj) &
                (df["pos_1b"] == pos)
                ]
            row = subset.sample(1)
            rows.append(row)
    return rows


def sample_df(hvm_dir, n_objs, grid_size, df_path=None):
    meta = HvMMetaData(hvm_dir)
    img_loader = HvMImageLoader(
        root_dir=hvm_dir,
        metadata=meta,
        preload_images=False,
    )
    img_loader.prepare_for_tasks(grid_size)
    df = img_loader.df
    if df_path is not None and os.path.isfile(df_path):
        print(f'Loading subset csv at {df_path}')
        df_subset = pd.read_csv(df_path)
    else:
        catid_to_positions = img_loader._task_cache.catid_to_positions
        chosen_objs = pick_objs(df, n_objs)
        rows = pick_locations(df, catid_to_positions, chosen_objs, loc_c=5)
        df_subset = pd.concat(rows, ignore_index=True)
        df_subset.to_csv(df_path)
        print(f'Saving subset csv at {df_path}')

    img_loader.df = df_subset
    img_loader._task_cache = None
    img_loader.prepare_for_tasks(grid_size)
    return img_loader, meta


def trial_to_row(
        task_config: TaskConfig,
        ds, emb, action, session_id,
        images_dir=Path('images'),
        bg_fp=Path('images') / 'gray_background.png',
        action_map={0: 'b', 1: 'x', 2: 'None'}
):
    zero_mask = torch.all(torch.isclose(
        emb,
        torch.tensor(0.0, dtype=emb.dtype)
    ), dim=1)
    nonzero_idx = torch.nonzero(~zero_mask).squeeze(1)

    subset = emb[nonzero_idx]
    img_mapper = HvMImageMapper(ds)
    files, decode_tuples = img_mapper._batch_decode_and_find(subset)
    fp_list = list()
    for i in range(task_config.frames_per_trial):
        if i in nonzero_idx:
            fp = files.pop(0)
            fp = Path(fp)
            fp = _subpath_after(
                fp, segment='HvM_with_discfade'
            )
            fp = fp.with_name(fp.name.lstrip("_"))
            fp = images_dir / fp
        else:
            fp = bg_fp
        fp_list.append(str(fp))

    row = {
        'session': session_id,
    }
    for i, (a, fp) in enumerate(zip(action, fp_list)):
        if i != 0 and a != 2:
            # only save action frames
            row[f'act{i + 1}'] = action_map[a.item()]

        row[f'stim{i + 1}'] = fp
    return row


def build_csv_from_dataset(
        task_config: TaskConfig,
        dataset,
) -> pd.DataFrame:
    rows = []
    session_id = 1
    for i, (emb, action, task_index) in enumerate(dataset):
        if i % task_config.trials_per_session == 0 and i > 0:
            session_id += 1
        rows.append(trial_to_row(
            task_config,
            dataset, emb, action, session_id
        ))
    stim_cols = task_config.stim_cols
    act_cols = task_config.act_cols
    df = pd.DataFrame(rows, columns=["session"] + stim_cols + act_cols)
    df.to_csv(task_config.task_trial_csv, index=False)
    print(f"Saved: {task_config.task_trial_csv} (rows={len(df)})")
    return df


def move_images_to_local_folder(df: pd.DataFrame, images_dir: Path) -> pd.DataFrame:
    project_dir = Path.cwd().parent
    images_dir = project_dir / 'resources' / images_dir
    shutil.rmtree(images_dir / 'Variation00_20110203', True)
    shutil.rmtree(images_dir / 'Variation03_20110128', True)
    shutil.rmtree(images_dir / 'Variation06_20110131', True)

    if 'stim1_fp' in df.columns and 'stim2_fp' in df.columns:
        fps = pd.concat([df['stim1_fp'], df['stim2_fp']]).unique()
    else:
        fps = df['filename'].unique()
    for fp in fps:
        fp = Path(fp)
        new_fp = _subpath_after(
            fp, segment='HvM_with_discfade'
        )
        new_fp = new_fp.with_name(new_fp.name.lstrip("_"))
        new_fp = images_dir / new_fp
        new_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fp, new_fp.absolute())
    return


def get_stim_pairs(
        task_config: TaskConfig,
        dataset: HvMTaskDataset,
        task_df: pd.DataFrame,
        stim_df: pd.DataFrame
):
    stim_pairs, features = list(), list()
    n_trials = len(task_df)
    action_map = {'b': 0, 'x': 1}
    stim_df['local_fp'] = stim_df['re_filename'].apply(
        lambda p: _subpath_after(Path(p), segment='HvM_with_discfade')
    )
    for i in range(n_trials):
        stims, categories, identities, positions, labels = list(), list(), list(), list(), list()
        for stim_col in task_config.stim_cols:
            stim_fp = task_df.loc[i, stim_col]
            stim_fp = _subpath_after(Path(stim_fp), segment='images')
            stim_row = stim_df[stim_df['local_fp'] == stim_fp]
            assert len(stim_row) == 1, f'{stim_fp} not found or multiple found in stim_df'
            row_idx = stim_row.index[0]
            stims.append(row_idx)
            categories.append(stim_df.loc[row_idx, 'cat_1b'])
            identities.append(stim_df.loc[row_idx, 'id_1b'])
            positions.append(stim_df.loc[row_idx, 'pos_1b'])
        for _ in range(len(task_config.stim_cols) - len(task_config.act_cols)):
            labels.append(2)
        for label_col in task_config.act_cols:
            labels.append(action_map[task_df.loc[i, label_col]])
        pairs, feature = dataset.get_stimuli_pair(
            {
                'row_idx': stims,
                'trial_label': labels,
                'task_id': TASK_NAME_TASK_INDEX[task_config.task_name],
                'category': categories,
                'identity': identities,
                'position': positions,
            },
        )
        stim_pairs.append(pairs)
        features.append(feature)
    return stim_pairs, features


def make_unordered_pair_df_with_features(stim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a stimulus dataframe build a dataframe of all unique unordered pairs
    and compute 3 binary features per pair.

    Columns in output:
        stim1, stim2 : row indices of the stimuli in stim_df
        same_category, same_obj, same_pos : example 0/1 features

    Customize the feature definitions as needed.
    """

    idx = stim_df.index.tolist()
    pairs = itertools.combinations(idx, 2)  # all i < j

    rows = []
    for i, j in pairs:
        s1 = stim_df.loc[i]
        s2 = stim_df.loc[j]

        same_category = int(s1["cat_1b"] == s2["cat_1b"])
        same_obj = int(s1["id_1b"] == s2["id_1b"])
        same_pos = int(s1["pos_1b"] == s2["pos_1b"])

        rows.append(
            {
                "stim1": i,
                "stim2": j,
                "same_category": same_category,
                "same_obj": same_obj,
                "same_pos": same_pos,
                "stim1_fp": s1["re_filename"],
                "stim2_fp": s2["re_filename"],
            }
        )

    pair_df = pd.DataFrame(rows)
    return pair_df


def make_ordered_pair_df_with_features(stim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a stimulus dataframe, build a dataframe of all ordered pairs (i -> j),
    i != j, and compute 3 binary features per pair.

    Output columns:
        stim1, stim2 : row indices of the stimuli in stim_df
        same_category, same_obj, same_pos : 0/1 features
        stim1_fp, stim2_fp : file paths for each stimulus
    """

    idx = stim_df.index.tolist()
    # all ordered pairs i -> j with i != j
    pairs = itertools.permutations(idx, 2)

    rows = []
    for i, j in pairs:
        s1 = stim_df.loc[i]
        s2 = stim_df.loc[j]

        same_category = int(s1["cat_1b"] == s2["cat_1b"])
        same_obj = int(s1["id_1b"] == s2["id_1b"])
        same_pos = int(s1["pos_1b"] == s2["pos_1b"])

        rows.append(
            {
                "stim1": i,
                "stim2": j,
                "same_category": same_category,
                "same_obj": same_obj,
                "same_pos": same_pos,
                "stim1_fp": s1["re_filename"],
                "stim2_fp": s2["re_filename"],
            }
        )

    pair_df = pd.DataFrame(rows)
    return pair_df


def select_balanced_pairs_from_stimuli(
        stim_df: pd.DataFrame,
        n_pairs: int = 100,
        feature_cols=("same_category", "same_obj", "same_pos"),
):
    """

    High-level pipeline:

    1. Build all unique unordered pairs of stimuli using row indices.
    2. Compute binary task features for each pair.
    3. Use ILP to select n_pairs with 50/50 split for each feature.

    This process can be used when n-back tasks are not used
    Returns:
        selected_pairs_df:
            columns: ['stim1', 'stim2'] + feature_cols
            (and optionally columns from stim_df if return_with_stim_info is True)
    """

    # build pair dataframe with features
    pair_df = make_unordered_pair_df_with_features(stim_df)

    # ILP setup
    df_bin = pair_df.copy()
    for f in feature_cols:
        df_bin[f] = df_bin[f].astype(int)

    idx = df_bin.index.tolist()
    prob = LpProblem("balanced_pair_selection", LpMinimize)

    # Decision variables: x_i ∈ {0,1} for each pair
    x = LpVariable.dicts("x", idx, lowBound=0, upBound=1, cat=LpBinary)

    # Objective: we only care about feasibility, so set objective to 0
    prob += 0

    # Constraint 1: exactly n_pairs selected
    prob += lpSum(x[i] for i in idx) == n_pairs, "num_pairs"

    # Constraint 2: for each feature, exactly half of the selected pairs must be True (1)
    half = n_pairs // 2
    for f in feature_cols:
        prob += lpSum(x[i] * df_bin.loc[i, f] for i in idx) == half, f"{f}_balance"

    # Solve
    prob.solve()
    status = LpStatus[prob.status]
    print("Solver status:", status)

    if status != "Optimal":
        raise RuntimeError("No feasible exact 50/50 solution found for these constraints.")

    chosen_idx = [i for i in idx if x[i].value() > 0.5]
    selected_pairs = pair_df.loc[chosen_idx].reset_index(drop=True)
    return selected_pairs


def make_chained_pair(
        pair_df: pd.DataFrame,
        n_trials: int,
        n_chains: int,
        feature_cols=("same_category", "same_obj", "same_pos"),
):
    """
    Vectorized version: Build chained pairs for n_trials with n_chains comparisons each.
    Attempts to balance multiple binary features approximately 50/50.

    Returns:
        trials: list[list[int]] of pair_df row indices, or None if stuck.
    """
    seen_unordered = set()
    trials = []

    # Track counts for each feature
    feature_cols = list(feature_cols)
    feature_counts = {f: 0 for f in feature_cols}
    total_pairs = n_trials * n_chains

    for t in range(n_trials):
        trial_idxs = []

        for k in range(n_chains):
            if k == 0:
                candidates = pair_df[~pair_df.apply(
                    lambda r: (min(r["stim1"], r["stim2"]),
                               max(r["stim1"], r["stim2"])) in seen_unordered,
                    axis=1
                )]
            else:
                prev_idx = trial_idxs[-1]
                s_prev2 = int(pair_df.loc[prev_idx, "stim2"])
                candidates = pair_df[pair_df["stim1"] == s_prev2].copy()
                candidates = candidates[~candidates.apply(
                    lambda r: (min(r["stim1"], r["stim2"]),
                               max(r["stim1"], r["stim2"])) in seen_unordered,
                    axis=1
                )]

            if len(candidates) == 0:
                return None  # stuck

            # Vectorized score computation
            n_current = len(trials) * n_chains + len(trial_idxs)
            feat_matrix = candidates[feature_cols].to_numpy()  # shape: (n_candidates, n_features)
            cur_ratios = np.array([feature_counts[f] / max(1, n_current) for f in feature_cols])
            new_ratios = (feat_matrix + cur_ratios * n_current) / (n_current + 1)
            scores = -np.abs(new_ratios - 0.5).sum(axis=1)

            # pick one candidate, weighted by score
            idx_pick = np.random.choice(
                candidates.index,
                p=(scores - scores.min() + 1e-6) / np.sum(scores - scores.min() + 1e-6)
            )
            s1 = int(pair_df.loc[idx_pick, "stim1"])
            s2 = int(pair_df.loc[idx_pick, "stim2"])
            seen_unordered.add((min(s1, s2), max(s1, s2)))
            trial_idxs.append(int(idx_pick))

            # update feature counts
            for i, f in enumerate(feature_cols):
                feature_counts[f] += pair_df.loc[idx_pick, f]

        trials.append(trial_idxs)

    return trials


def select_balanced_chained_pair(
        pair_df,
        n_trials=20,
        n_chains=5,
        n_tries_limit=10000,
):
    """
    Repeatedly try to build chained pairs and select the one with best balance.
    Used if 1back is included in human trials
    Args:
        pair_df:
        n_trials:
        n_chains:
        n_tries_limit:

    Returns:

    """
    n_tries = 0
    best_trial_df = None
    best_mean = np.array([float('inf'), float('inf'), float('inf')])
    while n_tries < n_tries_limit:
        if n_tries % 100 == 0:
            print(f'Try {n_tries}...')
        trials = make_chained_pair(pair_df, n_trials, n_chains)
        flat_trials = [idx for trial in trials for idx in trial]
        trial_df = pair_df.loc[flat_trials].reset_index(drop=True)
        cur_mean = np.array([
            trial_df['same_category'].mean(),
            trial_df['same_obj'].mean(),
            trial_df['same_pos'].mean(),
        ])
        if np.linalg.norm(cur_mean - 0.5) < np.linalg.norm(best_mean - 0.5):
            best_mean = cur_mean
            best_trial_df = trial_df
            print(f'New best mean: {best_mean} at try {n_tries}')
            if np.allclose(best_mean, 0.5):
                print('Perfect balance achieved!')
                break

        n_tries += 1
    return best_trial_df, trials
