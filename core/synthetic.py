"""Phase 0 검증용 synthetic perturbation 생성기.

분포에서 정상 벡터를 받아 known 방향으로 이동시켜 예측 가능한 오류 벡터를 만든다.
precision/recall 측정 → 분류기 sensitivity 검증에 사용.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from core.eojeol_vector import DIM_NAMES
from core.distribution import GaussianEojeolDistribution

PerturbationKind = Literal["rising", "falling", "flat", "elongation", "slow"]

# 각 kind가 classify()에서 기대하는 label
EXPECTED_LABEL: dict[str, str] = {
    "rising":      "rising 과도",
    "falling":     "falling 과도",
    "flat":        "억양 평탄",
    "elongation":  "마지막 음절 elongation",
    "slow":        "어절 전체 느림",
}

_DIM_IDX = {name: i for i, name in enumerate(DIM_NAMES)}


def perturb_vector(
    base: np.ndarray,
    dist: GaussianEojeolDistribution,
    kind: PerturbationKind,
    magnitude: float = 3.5,
) -> tuple[np.ndarray, str]:
    """기저 벡터에 perturbation을 주입해 known-label 벡터를 반환.

    Args:
        base: 정상 벡터 (12,). 분포 안쪽에 있어야 함.
        dist: 이미 fit()된 분포. mean/std 기준으로 perturbation 크기 결정.
        kind: 변형 종류.
        magnitude: perturbation 크기 (σ 단위).

    Returns:
        (perturbed_vector, expected_label) 튜플.
    """
    v = base.copy()
    std = dist.std_

    if kind == "rising":
        v[_DIM_IDX["f0_slope"]] = dist.mean_[_DIM_IDX["f0_slope"]] + magnitude * std[_DIM_IDX["f0_slope"]]

    elif kind == "falling":
        v[_DIM_IDX["f0_slope"]] = dist.mean_[_DIM_IDX["f0_slope"]] - magnitude * std[_DIM_IDX["f0_slope"]]

    elif kind == "flat":
        # slope을 평균 근방으로 고정 + f0_range를 -magnitude*σ
        slope_idx = _DIM_IDX["f0_slope"]
        range_idx = _DIM_IDX["f0_range"]
        v[slope_idx] = dist.mean_[slope_idx]                            # |z_slope| < 0.5 보장
        v[range_idx] = dist.mean_[range_idx] - magnitude * std[range_idx]

    elif kind == "elongation":
        idx = _DIM_IDX["last_syl_ratio"]
        v[idx] = dist.mean_[idx] + magnitude * std[idx]

    elif kind == "slow":
        idx = _DIM_IDX["duration"]
        v[idx] = dist.mean_[idx] + magnitude * std[idx]

    return v, EXPECTED_LABEL[kind]


def generate_test_cases(
    dist: GaussianEojeolDistribution,
    n_per_class: int = 10,
    magnitude: float = 3.5,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, str]]:
    """각 perturbation 종류별 n_per_class개의 (vector, label) 쌍 생성.

    base 벡터는 분포 mean 주변 0.3σ 내 랜덤 샘플 → 실제 검증 다양성 확보.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    cases: list[tuple[np.ndarray, str]] = []
    for kind in EXPECTED_LABEL:
        for _ in range(n_per_class):
            # 분포 내부의 무작위 정상 벡터에서 시작
            base = dist.mean_ + rng.standard_normal(_N_DIMS) * dist.std_ * 0.3
            v, label = perturb_vector(base, dist, kind, magnitude=magnitude)  # type: ignore[arg-type]
            cases.append((v, label))

    return cases


# ── 개수 상수 (DIM_NAMES에서 동기화) ────────────────────────────────────────
_N_DIMS = len(DIM_NAMES)