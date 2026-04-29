import numpy as np
from dataclasses import dataclass
from dtaidistance import dtw
from core.f0_extractor import extract_f0


@dataclass
class ComparisonResult:
    aligned_native: np.ndarray
    aligned_learner: np.ndarray
    native_indices: np.ndarray    # DTW 각 위치의 원본 native 프레임 인덱스
    learner_indices: np.ndarray   # DTW 각 위치의 원본 learner 프레임 인덱스
    voiced_mask: np.ndarray       # 양쪽 모두 voiced인 DTW 위치
    native_times: np.ndarray      # native 원본 시간 배열 (음절 경계 매핑용)
    learner_times: np.ndarray     # learner 원본 시간 배열 (플롯 등 참고용)


class IntonationComparator:
    def compare(self, native_path: str, learner_path: str) -> ComparisonResult:
        native = extract_f0(native_path)
        learner = extract_f0(learner_path)

        path = dtw.warping_path(native.f0, learner.f0)
        native_idx = np.array([i for i, j in path])
        learner_idx = np.array([j for i, j in path])

        return ComparisonResult(
            aligned_native=native.f0[native_idx],
            aligned_learner=learner.f0[learner_idx],
            native_indices=native_idx,
            learner_indices=learner_idx,
            voiced_mask=native.voiced_mask[native_idx] & learner.voiced_mask[learner_idx],
            native_times=native.times,
            learner_times=learner.times,
        )