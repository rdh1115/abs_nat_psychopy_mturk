import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info, default_collate

from task.base import HvMTaskDataset
from task.dms import DMSDataset, CtxDMSDataset, InterDMSDataset
from task.nback import NBackDataset
from utils.stim_io import HvMImageLoader, HvMImageMapper

TASK_NAME_TASK_INDEX = {
    '1back_position': 1,
    '1back_identity': 2,
    '1back_category': 3,
    '2back_position': 4,
    '2back_identity': 5,
    '2back_category': 6,
    '3back_position': 7,
    '3back_identity': 8,
    '3back_category': 9,

    'interdms_AABB_position_position': 10,
    'interdms_AABB_position_identity': 11,
    'interdms_AABB_position_category': 12,
    'interdms_AABB_identity_position': 13,
    'interdms_AABB_identity_identity': 14,
    'interdms_AABB_identity_category': 15,
    'interdms_AABB_category_position': 16,
    'interdms_AABB_category_identity': 17,
    'interdms_AABB_category_category': 18,
    'interdms_ABAB_position_position': 19,
    'interdms_ABAB_position_identity': 20,
    'interdms_ABAB_position_category': 21,
    'interdms_ABAB_identity_position': 22,
    'interdms_ABAB_identity_identity': 23,
    'interdms_ABAB_identity_category': 24,
    'interdms_ABAB_category_position': 25,
    'interdms_ABAB_category_identity': 26,
    'interdms_ABAB_category_category': 27,
    'interdms_ABBA_position_position': 28,
    'interdms_ABBA_position_identity': 29,
    'interdms_ABBA_position_category': 30,
    'interdms_ABBA_identity_position': 31,
    'interdms_ABBA_identity_identity': 32,
    'interdms_ABBA_identity_category': 33,
    'interdms_ABBA_category_position': 34,
    'interdms_ABBA_category_identity': 35,
    'interdms_ABBA_category_category': 36,

    'ctxdms_position_category_identity': 37,
    'ctxdms_position_identity_category': 38,
    'ctxdms_identity_position_category': 39,
    'ctxdms_category_identity_position': 40,

    'dms_position': 41,
    'dms_identity': 42,
    'dms_category': 43
}

TASK_DATALOADERS = {
    'dms': {
        'dms_category': (DMSDataset, {"feature": "category"}),
        'dms_identity': (DMSDataset, {"feature": "identity"}),
        'dms_position': (DMSDataset, {"feature": "position"}),
    },
    'nback': {
        '1back_category': (NBackDataset, {"feature": "category", "nback_n": 1}),
        '2back_category': (NBackDataset, {"feature": "category", "nback_n": 2}),
        '3back_category': (NBackDataset, {"feature": "category", "nback_n": 3}),

        '1back_position': (NBackDataset, {"feature": "position", "nback_n": 1}),
        '2back_position': (NBackDataset, {"feature": "position", "nback_n": 2}),
        '3back_position': (NBackDataset, {"feature": "position", "nback_n": 3}),

        '1back_identity': (NBackDataset, {"feature": "identity", "nback_n": 1}),
        '2back_identity': (NBackDataset, {"feature": "identity", "nback_n": 2}),
        '3back_identity': (NBackDataset, {"feature": "identity", "nback_n": 3}),
    },
    'interdms': {
        'interdms_AABB_category_category': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'category'}),
        'interdms_AABB_category_identity': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'identity'}),
        'interdms_AABB_category_position': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'position'}),
        'interdms_AABB_identity_category': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'category'}),
        'interdms_AABB_identity_identity': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'identity'}),
        'interdms_AABB_identity_position': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'position'}),
        'interdms_AABB_position_category': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'category'}),
        'interdms_AABB_position_identity': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'identity'}),
        'interdms_AABB_position_position': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'position'}),
        'interdms_ABAB_category_category': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'category'}),
        'interdms_ABAB_category_identity': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'identity'}),
        'interdms_ABAB_category_position': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'position'}),
        'interdms_ABAB_identity_category': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'category'}),
        'interdms_ABAB_identity_identity': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'identity'}),
        'interdms_ABAB_identity_position': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'position'}),
        'interdms_ABAB_position_category': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'category'}),
        'interdms_ABAB_position_identity': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'identity'}),
        'interdms_ABAB_position_position': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'position'}),
        'interdms_ABBA_category_category': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'category'}),
        'interdms_ABBA_category_identity': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'identity'}),
        'interdms_ABBA_category_position': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'position'}),
        'interdms_ABBA_identity_category': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'category'}),
        'interdms_ABBA_identity_identity': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'identity'}),
        'interdms_ABBA_identity_position': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'position'}),
        'interdms_ABBA_position_category': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'category'}),
        'interdms_ABBA_position_identity': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'identity'}),
        'interdms_ABBA_position_position': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'position'}),
    },
    'ctxdms': {
        'ctxdms_category_identity_position': (
            CtxDMSDataset,
            {"features": ["category", "identity", "position"]}
        ),
        'ctxdms_position_category_identity': (
            CtxDMSDataset,
            {"features": ["position", "category", "identity"]}
        ),
        'ctxdms_position_identity_category': (
            CtxDMSDataset,
            {"features": ["position", "identity", "category"]}
        ),
        'ctxdms_identity_position_category': (
            CtxDMSDataset,
            {"features": ["identity", "position", "category"]}
        )
    }
}

TASK_IDX_TASK_TYPE = dict()
for i in range(1, 44):
    if i < 10:
        TASK_IDX_TASK_TYPE[i] = NBackDataset
    elif i < 37:
        TASK_IDX_TASK_TYPE[i] = InterDMSDataset
    elif i < 41:
        TASK_IDX_TASK_TYPE[i] = CtxDMSDataset
    else:
        TASK_IDX_TASK_TYPE[i] = DMSDataset

TASK_TYPE_DICT = dict()
for i in range(1, 44):
    if i < 10:
        TASK_TYPE_DICT[i] = 'NBack'
    elif i < 37:
        TASK_TYPE_DICT[i] = 'InterDms'
    elif i < 41:
        TASK_TYPE_DICT[i] = 'CtxDms'
    else:
        TASK_TYPE_DICT[i] = 'Dms'

# Training constants
RNN_TYPES = ['GRU', 'LSTM', 'RNN']
INPUT_SIZE = 8 + 64 + 9 + 3 + 1 + 43  # 8 categories + 64 objects + 9 positions + 3 view + 1 size + 43 tasks = 128
OUTPUT_SIZE = 3


def filter_tasks(tasks):
    FLAT_TASKS = {
        task_name: task_def
        for task_group in TASK_DATALOADERS.values()
        for task_name, task_def in task_group.items()
    }

    def _filter_single_task(task_str):
        # Task group (e.g., "dms")
        if task_str in TASK_DATALOADERS:
            return TASK_DATALOADERS[task_str]

        # Task name (e.g., "1back_identity")
        elif task_str in FLAT_TASKS:
            return {task_str: FLAT_TASKS[task_str]}

        # Feature name (e.g., "category", "identity", "position")
        elif task_str in ['category', 'identity', 'position']:
            task_by_feature = {}

            for task_group in TASK_DATALOADERS.values():
                for task_name, (cls, kwargs) in task_group.items():
                    if "feature" in kwargs and kwargs["feature"] == task_str:
                        task_by_feature[task_name] = (cls, kwargs)
                    elif "feature_1" in kwargs and "feature_2" in kwargs:
                        if kwargs["feature_1"] == kwargs["feature_2"] == task_str:
                            task_by_feature[task_name] = (cls, kwargs)

            return task_by_feature

        else:
            raise ValueError(f"Unknown task string: {task_str}, must be one of "
                             f"[all, task group names, task names, or features]")

    if tasks is None or tasks == 'all' or tasks == TASK_DATALOADERS:
        return FLAT_TASKS
    elif isinstance(tasks, str):
        return _filter_single_task(tasks)
    elif isinstance(tasks, list):
        filtered = {}
        TASK_INDEX_TASK_NAME = {idx: task for task, idx in TASK_NAME_TASK_INDEX.items()}

        for task in tasks:
            # If int: assume it's a task index
            if isinstance(task, int):
                task_name = TASK_INDEX_TASK_NAME[task]
                filtered[task_name] = FLAT_TASKS[task_name]

            # If string: filter using base logic
            elif isinstance(task, str):
                partial = _filter_single_task(task)
                filtered.update(partial)

            else:
                raise ValueError("Tasks in list must be str (task group/name/feature) or int (index).")

        return filtered
    else:
        raise TypeError(f"tasks must be None, str, or list of str/int. got {type(tasks)} instead")


def abstract_collate(batch, task_dataset):
    inputs, labels, task_indices = zip(*batch)
    task_info = [
        {
            i: task_dataset.decode_embedding(emb) if not np.allclose(emb, 0) else None
            for i, emb in enumerate(batch)
        }
        for batch in inputs
    ]
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    task_indices = torch.stack(task_indices, dim=0)
    return inputs, labels, task_indices, task_info


def natural_collate(batch, mapper: HvMImageMapper):
    # batch is list of (emb_arr, actions, task_idx)
    emb, actions, task_idx = zip(*batch)
    emb = torch.stack(emb)  # (B, T, D)
    imgs, task_info = mapper.map_batch(emb)  # img tensor shape (B, T, C, H, W)
    actions = default_collate(actions).long()  # torch tensor (B, T)
    task_idx = default_collate(task_idx).long()  # torch tensor (B,)
    return imgs, actions, task_idx, task_info


class NaturalCollateFn:
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, batch):
        return natural_collate(batch, self.mapper)


class AbstractCollateFn:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        return abstract_collate(batch, self.dataset)


class MultiTaskDataset(HvMTaskDataset, IterableDataset):
    def __init__(
            self,
            hvm_loader,
            dataloaders_dict=None,
            mode='train',
            dataset_size=128,
            task_weights=None,
    ):
        """
        Initializes the MultiTaskDataset.

        Args:
            dataloaders_dict (dict): Dictionary containing task names as keys and tuples of (DatasetClass, kwargs) as values.
            mode (str): 'train' or 'val' to indicate the dataset split.
        """
        if dataloaders_dict is None:
            dataloaders_dict = TASK_DATALOADERS
        super().__init__(
            hvm_loader=hvm_loader,
            dataset_size=dataset_size,
            task_len=6,
        )
        # multitask datasets use self.datasets to create trials
        self.dataset = None

        if dataloaders_dict is None:
            dataloaders_dict = TASK_DATALOADERS

        self.datasets = dict()
        self.task_names = list(dataloaders_dict.keys())
        self.dataloaders_dict = dataloaders_dict

        self.task_dataset_size = self.get_dataset_size_per_task(dataset_size)
        for task_name, (DatasetClass, kwargs) in dataloaders_dict.items():
            dataset_kwargs = kwargs.copy()
            dataset_kwargs.update({
                'hvm_loader': hvm_loader,
                'pad_to': 6,
                'dataset_size': self.task_dataset_size[task_name],
            })
            # Instantiate the dataset
            self.datasets[task_name] = DatasetClass(**dataset_kwargs)

        # Calculate the maximum length among all datasets
        sizes = {name: len(ds) for name, ds in self.datasets.items()}
        assert sum(sizes.values()) == dataset_size, "Subtask sizes must sum to total"
        # Calculate weights based on dataset sizes for balanced sampling
        if task_weights is None:
            task_weights = {name: size / dataset_size for name, size in sizes.items()}
        self.task_weights = task_weights

        if 'val' in mode:
            self.mode = 'val'
        else:
            self.mode = mode
        if self.mode == 'val':
            # reset once
            for ds in self.datasets.values():
                if hasattr(ds, 'reset'):
                    ds.reset()
            # self.trials = self.build_dataset()
            self.trial_pairs = self.build_pairs()

    def get_dataset_size_per_task(self, total_dataset_size):
        """
        Evenly distribute total_dataset_size across task groups (e.g., 'dms', 'nback'),
        then evenly among subtasks within each group.
        """

        # TODO: do we want to do this? this could cause duplicate task trials
        # Group tasks by task group prefix (e.g., 'dms', 'nback', etc.)
        task_group_dict = {task_group: dict() for task_group in TASK_DATALOADERS.keys()}
        task_groups = set()
        for task_name, v in self.dataloaders_dict.items():
            task_group = task_name.split('_')[0]
            if 'back' in task_group:
                task_group = 'nback'
            task_group_dict[task_group][task_name] = v
            task_groups.add(task_group)
        task_group_dict = {k: v for k, v in task_group_dict.items() if k in task_groups}

        num_groups = len(task_groups)
        base_group_size = total_dataset_size // num_groups
        group_remainder = total_dataset_size % num_groups

        task_dataset_size = {task: 0 for task in self.dataloaders_dict.keys()}
        for i, (group, subtasks) in enumerate(sorted(task_group_dict.items())):
            group_size = base_group_size + (1 if i < group_remainder else 0)
            num_subtasks = len(subtasks)
            base_subtask_size = group_size // num_subtasks
            subtask_remainder = group_size % num_subtasks

            for j, task_name in enumerate(subtasks):
                task_size = base_subtask_size + (1 if j < subtask_remainder else 0)
                task_dataset_size[task_name] = task_size
        return task_dataset_size

    def build_dataset(self):
        embs, actions, indices = [], [], []

        ds: HvMTaskDataset
        total = 0
        for task_name, ds in self.datasets.items():
            assert ds.dataset.shape[0] == ds.actions.shape[0]
            subtask_size = ds.dataset.shape[0]
            embs.append(ds.dataset)
            actions.append(ds.actions)
            indices.append(torch.tensor(ds.task_index).expand(subtask_size, -1))
            total += subtask_size
        assert total == self.dataset_size
        embs = torch.cat(embs, dim=0)  # [N, ...]
        actions = torch.cat(actions, dim=0)  # [N, ...]
        indices = torch.cat(indices, dim=0)  # [N, 43]
        return embs, actions, indices

    def build_pairs(self):
        flat = list()
        for task_name, ds in self.datasets.items():
            n = len(ds)
            idxs = np.arange(n)
            flat.extend((task_name, int(i)) for i in idxs)
        return flat

    def __len__(self):
        return self.dataset_size

    # def __iter__(self):
    #     worker_info = get_worker_info()
    #
    #     if not self.mode == 'val':
    #         # regenerate at the start of each epoch
    #         for ds in self.datasets.values():
    #             ds.reset()
    #         embs, actions, indices = self.build_dataset()
    #     else:
    #         embs, actions, indices = self.trials
    #
    #     N = embs.shape[0]
    #     g = torch.Generator(device='cpu')
    #     g.manual_seed(torch.initial_seed() % (2 ** 63 - 1))
    #     perm = torch.randperm(N, generator=g)
    #     if worker_info is None:
    #         selected = perm
    #     else:
    #         n_worker = worker_info.num_workers
    #         w_id = worker_info.id
    #         base = N // n_worker
    #         r = N % n_worker
    #         start = w_id * base + min(w_id, r)
    #         end = start + base + (1 if w_id < r else 0)
    #         selected = perm[start:end]
    #
    #     for k in selected.tolist():
    #         yield embs[k], actions[k], indices[k]
    #     if self.mode != 'val':
    #         del embs, actions, indices, selected
    #     return

    def __iter__(self):
        wi = get_worker_info()

        if self.mode != 'val':
            for ds in self.datasets.values():
                if hasattr(ds, 'reset'):
                    ds.reset()
            flat = self.build_pairs()
        else:
            flat = self.trial_pairs

        N = len(flat)
        g = torch.Generator(device='cpu')
        g.manual_seed(torch.initial_seed() % (2 ** 63 - 1))
        perm = torch.randperm(N, generator=g).tolist()

        if wi is not None:
            w, i = wi.num_workers, wi.id
            base, r = N // w, N % w
            start = i * base + min(i, r)
            end = start + base + (1 if i < r else 0)
            perm = perm[start:end]

        for k in perm:
            task_name, idx = flat[k]
            yield self.datasets[task_name][idx]

    @staticmethod
    def get_stimuli_pair(trial_info):
        task_id = trial_info['task_id']
        return TASK_IDX_TASK_TYPE[task_id].get_stimuli_pair(trial_info)
