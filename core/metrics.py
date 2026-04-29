import numpy as np
from dataclasses import dataclass
from core.comparator import SyllableComparison


@dataclass
class SyllableMetrics:
    syllable_idx: int
    rmse: float
    pearson: float          # nan if voiced_count < 5
    slope_diff: float       # native_slope - learner_slope, nan if voiced_count < 3
    voiced_frame_count: int
    duration_ratio: float   # learner_duration / native_duration


def compute_metrics(comparisons: list[SyllableComparison]) -> list[SyllableMetrics]:
    results = []
    for c in comparisons:
        v = c.voiced_mask
        count = int(v.sum())
        duration_ratio = (
            c.learner_duration / c.native_duration if c.native_duration > 0 else float("nan")
        )

        if count == 0:
            results.append(SyllableMetrics(
                syllable_idx=c.syllable_idx,
                rmse=float("nan"),
                pearson=float("nan"),
                slope_diff=float("nan"),
                voiced_frame_count=0,
                duration_ratio=duration_ratio,
            ))
            continue

        native_v = c.native_f0[v]
        learner_v = c.learner_f0[v]

        rmse = float(np.sqrt(np.mean((native_v - learner_v) ** 2)))

        pearson = float(np.corrcoef(native_v, learner_v)[0, 1]) if count >= 5 else float("nan")

        if count >= 3:
            positions = np.where(v)[0].astype(float)
            native_slope = float(np.polyfit(positions, native_v, 1)[0])
            learner_slope = float(np.polyfit(positions, learner_v, 1)[0])
            slope_diff = native_slope - learner_slope
        else:
            slope_diff = float("nan")

        results.append(SyllableMetrics(
            syllable_idx=c.syllable_idx,
            rmse=rmse,
            pearson=pearson,
            slope_diff=slope_diff,
            voiced_frame_count=count,
            duration_ratio=duration_ratio,
        ))
    return results