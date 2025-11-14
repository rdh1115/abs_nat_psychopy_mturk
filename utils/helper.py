from pathlib import Path
from typing import Optional

import numpy as np


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


def mturk_localize(resource_dir, p: str):
    segment = "HvM_with_discfade"
    relative = _subpath_after(Path(p), segment)
    relative = relative.with_name(relative.name.lstrip("_"))
    return resource_dir / relative
