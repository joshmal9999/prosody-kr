import numpy as np
from dataclasses import dataclass


@dataclass
class SyllableRMSE:
    syllable_idx: int
    t_start: float
    t_end: float
    rmse: float           # voiced frame 없으면 nan
    voiced_frame_count: int


def rmse_by_syllable(
    aligned_native: np.ndarray,
    aligned_learner: np.ndarray,
    learner_indices: np.ndarray,
    voiced_mask: np.ndarray,
    learner_times: np.ndarray,
    syllable_boundaries: list[tuple[float, float]],  # [(t_start, t_end), ...]
) -> list[SyllableRMSE]:
    results = []
    learner_times_aligned = learner_times[learner_indices]

    for idx, (t_start, t_end) in enumerate(syllable_boundaries):
        in_syllable = (learner_times_aligned >= t_start) & (learner_times_aligned < t_end)
        mask = in_syllable & voiced_mask
        count = int(mask.sum())
        if count == 0:
            results.append(SyllableRMSE(idx, t_start, t_end, float("nan"), 0))
            continue
        rmse = float(np.sqrt(np.mean((aligned_native[mask] - aligned_learner[mask]) ** 2)))
        results.append(SyllableRMSE(idx, t_start, t_end, rmse, count))
    return results