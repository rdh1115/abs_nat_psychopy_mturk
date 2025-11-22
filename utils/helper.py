from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np


@dataclass
class TaskConfig:
    task_name: str
    features: List[str]

    act_cols: List[str]  # e.g. ["act3", "act4"]
    stim_cols: List[str]  # e.g. ["stim1", "stim2", "stim3", "stim4"]
    act_indices: List[int]

    task_trial_csv: str
    task_meta_csv: str

    num_sessions: int = 5
    trials_per_session: int = 20
    frames_per_trial: int = 6

    resp_col: str = "resp.keys"  # e.g. "resp.keys"
    rt_col: str = "resp.rt"  # e.g. "resp.rt" or None if no RT
    trial_start_col: str = "TrialIntro.started"  # e.g. "TrialIntro.started"


def get_one_hot(index: int, total: int = 43):
    result = np.zeros(total)
    result[index - 1] = 1
    return result


def _subpath_after(p: Path, segment: str) -> Optional[Path]:
    """Return the subpath (as Path) after `segment`"""
    try:
        idx = p.parts.index(segment)
    except ValueError:
        return None
    return Path(*p.parts[idx + 1:])
