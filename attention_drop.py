import numpy as np
from typing import List, Dict, Optional


def find_attention_drop_point(session_data: List[Dict], threshold: float = 0.5) -> Optional[Dict]:
    """Find the moment where engagement experienced the biggest drop.

    Parameters
    ----------
    session_data : List[Dict]
        List containing session entries with at least a ``score`` and ``timestamp`` field.
    threshold : float, optional
        Minimum score difference considered a drop. Defaults to ``0.5``.

    Returns
    -------
    Optional[Dict]
        ``None`` if no drop could be determined, otherwise a dictionary with the
        keys ``index``, ``timestamp`` and ``score`` of the drop moment.
    """
    if not session_data or len(session_data) < 2:
        return None

    scores = [point.get("score", 0) for point in session_data]
    diffs = np.diff(scores)
    drop_index = int(np.argmin(diffs)) + 1  # index after the drop
    drop_value = diffs[drop_index - 1]

    if drop_value < -threshold:
        drop_point = session_data[drop_index]
    else:
        # no significant drop, use lowest scoring moment
        drop_index = int(np.argmin(scores))
        drop_point = session_data[drop_index]

    return {
        "index": drop_index,
        "timestamp": drop_point.get("timestamp"),
        "score": drop_point.get("score"),
    }
