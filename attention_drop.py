import numpy as np
from typing import List, Dict, Optional


def find_attention_drop_moment(session_data: List[Dict], *, method: str = "min", threshold: float = 0.5) -> Optional[Dict]:
    """Return information about the moment where engagement dropped.

    Parameters
    ----------
    session_data : list of dict
        Collection of data points with a ``score`` field and ``timestamp``.
    method : {"min", "largest-drop"}
        How to determine the drop moment. ``"min"`` returns the lowest score
        in the session while ``"largest-drop"`` looks for the biggest
        decrease between consecutive scores.
    threshold : float
        Minimum drop amount when ``method`` is ``"largest-drop"``.

    Returns
    -------
    dict or None
        A dictionary with ``index``, ``timestamp`` and ``score`` of the drop
        moment, or ``None`` if it cannot be determined.
    """
    if not session_data:
        return None

    scores = [d.get("score", 0) for d in session_data]

    if method == "largest-drop" and len(scores) > 1:
        diffs = np.diff(scores)
        drop_idx = int(np.argmin(diffs))
        if diffs[drop_idx] < -abs(threshold):
            idx = drop_idx + 1
        else:
            idx = int(np.argmin(scores))
    else:
        idx = int(np.argmin(scores))

    entry = session_data[idx]
    return {"index": idx, "timestamp": entry.get("timestamp"), "score": entry.get("score")}
