"""Phase 0: framework 검증 + synthetic perturbation precision/recall ≥ 0.7."""
from __future__ import annotations

import numpy as np
import pytest

from core.distribution import GaussianEojeolDistribution, _MIN_SAMPLES_FULL_COV
from core.eojeol_vector import DIM_NAMES, _N_DIMS, extract_eojeol_vector
from core.f0_extractor import F0Result
from core.synthetic import (
    EXPECTED_LABEL,
    PerturbationKind,
    generate_test_cases,
    perturb_vector,
)

# ── 공통 fixture ─────────────────────────────────────────────────────────────

_RNG = np.random.default_rng(42)

_DIST_MEAN = np.array([
    0.0,  # f0_mean
    0.5,  # f0_std
    0.1,  # f0_slope
    1.2,  # f0_range
    0.5,  # f0_max_pos
    0.3,  # f0_min_pos
    -0.1, # f0_start
    0.1,  # f0_end
    0.5,  # duration
    2.5,  # syllable_count
    1.0,  # last_syl_ratio
    0.7,  # voiced_ratio
])

_DIST_STD = np.array([
    0.5, 0.2, 0.3, 0.4,
    0.2, 0.2, 0.4, 0.4,
    0.1, 0.5, 0.3, 0.1,
])


def _make_dist(n: int = 40) -> GaussianEojeolDistribution:
    vectors = _RNG.normal(_DIST_MEAN, _DIST_STD, size=(n, _N_DIMS))
    dist = GaussianEojeolDistribution()
    dist.fit(vectors)
    return dist


def _make_f0_result(
    total_frames: int = 60,
    voiced_frac: float = 0.7,
    rng: np.random.Generator | None = None,
) -> F0Result:
    if rng is None:
        rng = np.random.default_rng(1)
    times = np.linspace(0.0, 1.0, total_frames)
    voiced_mask = rng.random(total_frames) < voiced_frac
    f0_raw = np.where(voiced_mask, rng.uniform(100, 300, total_frames), 0.0)
    # 간단한 z-score 정규화
    v = f0_raw[voiced_mask]
    if len(v) >= 2:
        f0 = np.where(voiced_mask, (f0_raw - v.mean()) / max(v.std(), 1e-8), 0.0)
    else:
        f0 = np.zeros_like(f0_raw)
    return F0Result(f0=f0, f0_raw=f0_raw, voiced_mask=voiced_mask,
                    times=times, sampling_rate=16000.0)


# ── TestEojeolVectorShape ────────────────────────────────────────────────────

class TestEojeolVector:
    def test_output_shape(self):
        f0 = _make_f0_result()
        syl_b = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        v = extract_eojeol_vector(f0, (0.0, 1.0), syl_b)
        assert v.shape == (_N_DIMS,)
        assert v.dtype == np.float64

    def test_dim_names_count(self):
        assert len(DIM_NAMES) == _N_DIMS == 12

    def test_duration_matches_boundary(self):
        f0 = _make_f0_result()
        v = extract_eojeol_vector(f0, (0.2, 0.7), [(0.2, 0.45), (0.45, 0.7)])
        assert v[DIM_NAMES.index("duration")] == pytest.approx(0.5, abs=1e-9)

    def test_syllable_count_matches(self):
        f0 = _make_f0_result()
        syl_b = [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]
        v = extract_eojeol_vector(f0, (0.0, 1.0), syl_b)
        assert v[DIM_NAMES.index("syllable_count")] == pytest.approx(3.0)

    def test_voiced_ratio_in_range(self):
        f0 = _make_f0_result(voiced_frac=0.6)
        v = extract_eojeol_vector(f0, (0.0, 1.0), [(0.0, 0.5), (0.5, 1.0)])
        assert 0.0 <= v[DIM_NAMES.index("voiced_ratio")] <= 1.0

    def test_no_voiced_frames_returns_zeros(self):
        f0 = _make_f0_result()
        # voiced_mask를 모두 False로
        f0 = F0Result(
            f0=np.zeros(60), f0_raw=np.zeros(60),
            voiced_mask=np.zeros(60, dtype=bool),
            times=f0.times, sampling_rate=16000.0,
        )
        v = extract_eojeol_vector(f0, (0.0, 1.0), [(0.0, 0.5), (0.5, 1.0)])
        assert v[DIM_NAMES.index("f0_mean")] == pytest.approx(0.0)
        assert v[DIM_NAMES.index("f0_std")] == pytest.approx(0.0)

    def test_last_syl_ratio_equals_one_for_single_syllable(self):
        f0 = _make_f0_result()
        v = extract_eojeol_vector(f0, (0.0, 1.0), [(0.0, 1.0)])
        assert v[DIM_NAMES.index("last_syl_ratio")] == pytest.approx(1.0)


# ── TestDistributionFit ──────────────────────────────────────────────────────

class TestDistributionFit:
    def test_fit_full_cov(self):
        dist = _make_dist(n=40)
        assert dist._mode == "full_shrunk"
        assert dist.cov_inv_.shape == (_N_DIMS, _N_DIMS)

    def test_fit_diagonal_fallback(self):
        vectors = _RNG.standard_normal((8, _N_DIMS))
        dist = GaussianEojeolDistribution()
        dist.fit(vectors)
        assert dist._mode == "diagonal"

    def test_mean_in_distribution(self):
        dist = _make_dist()
        d = dist.mahalanobis(dist.mean_)
        assert d == pytest.approx(0.0, abs=1e-9)

    def test_mahalanobis_far_vector(self):
        dist = _make_dist()
        far = dist.mean_ + 10 * dist.std_
        assert dist.mahalanobis(far) > 2.5

    def test_per_dim_z_shape(self):
        dist = _make_dist()
        z = dist.per_dim_z(dist.mean_)
        assert set(z.keys()) == set(DIM_NAMES)

    def test_per_dim_z_mean_is_zero(self):
        dist = _make_dist()
        z = dist.per_dim_z(dist.mean_)
        for name, val in z.items():
            assert val == pytest.approx(0.0, abs=1e-9), name

    def test_is_in_distribution_true_for_mean(self):
        dist = _make_dist()
        assert dist.is_in_distribution(dist.mean_)

    def test_is_out_of_distribution_for_extreme(self):
        dist = _make_dist()
        far = dist.mean_ + 10 * dist.std_
        assert not dist.is_in_distribution(far)

    def test_data_quality_warning_when_few_samples(self):
        vectors = _RNG.standard_normal((5, _N_DIMS))
        dist = GaussianEojeolDistribution()
        dist.fit(vectors)
        dq = dist.data_quality()
        assert dq.warning is not None
        assert dq.covariance_mode == "diagonal"

    def test_data_quality_no_warning_enough_samples(self):
        dist = _make_dist(n=30)
        dq = dist.data_quality()
        assert dq.warning is None


# ── TestClassifyRules ────────────────────────────────────────────────────────

class TestClassifyRules:
    def _dist_and_mean(self) -> tuple[GaussianEojeolDistribution, np.ndarray]:
        dist = _make_dist()
        return dist, dist.mean_.copy()

    def test_normal_vector_no_labels(self):
        dist, v = self._dist_and_mean()
        assert dist.classify(v) == []

    def test_rising_detected(self):
        dist, v = self._dist_and_mean()
        v[DIM_NAMES.index("f0_slope")] = dist.mean_[DIM_NAMES.index("f0_slope")] + 3.0 * dist.std_[DIM_NAMES.index("f0_slope")]
        assert "rising 과도" in dist.classify(v)

    def test_falling_detected(self):
        dist, v = self._dist_and_mean()
        v[DIM_NAMES.index("f0_slope")] = dist.mean_[DIM_NAMES.index("f0_slope")] - 3.0 * dist.std_[DIM_NAMES.index("f0_slope")]
        assert "falling 과도" in dist.classify(v)

    def test_flat_detected(self):
        dist, v = self._dist_and_mean()
        slope_idx = DIM_NAMES.index("f0_slope")
        range_idx = DIM_NAMES.index("f0_range")
        v[slope_idx] = dist.mean_[slope_idx]
        v[range_idx] = dist.mean_[range_idx] - 3.5 * dist.std_[range_idx]
        assert "억양 평탄" in dist.classify(v)

    def test_elongation_detected(self):
        dist, v = self._dist_and_mean()
        idx = DIM_NAMES.index("last_syl_ratio")
        v[idx] = dist.mean_[idx] + 3.0 * dist.std_[idx]
        assert "마지막 음절 elongation" in dist.classify(v)

    def test_slow_detected(self):
        dist, v = self._dist_and_mean()
        idx = DIM_NAMES.index("duration")
        v[idx] = dist.mean_[idx] + 3.0 * dist.std_[idx]
        assert "어절 전체 느림" in dist.classify(v)


# ── TestPrecisionRecall (통과 기준: precision/recall ≥ 0.7) ──────────────────

class TestPrecisionRecall:
    def _compute_metrics(
        self,
        n_samples: int = 40,
        n_per_class: int = 20,
        magnitude: float = 3.5,
    ) -> tuple[float, float]:
        dist = _make_dist(n=n_samples)
        cases = generate_test_cases(dist, n_per_class=n_per_class, magnitude=magnitude)

        tp = fp = fn = 0
        for v, expected_label in cases:
            predicted = dist.classify(v)
            if expected_label in predicted:
                tp += 1
            else:
                fn += 1
            # 예상치 못한 레이블도 fp로 계산
            fp += sum(1 for p in predicted if p != expected_label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall

    def test_recall_above_threshold(self):
        _, recall = self._compute_metrics()
        assert recall >= 0.7, f"recall={recall:.3f} < 0.7"

    def test_precision_above_threshold(self):
        precision, _ = self._compute_metrics()
        assert precision >= 0.7, f"precision={precision:.3f} < 0.7"

    def test_per_kind_recall(self):
        """각 perturbation 종류별 recall ≥ 0.5 (개별 상세 진단)."""
        dist = _make_dist(n=40)
        rng = np.random.default_rng(99)
        n = 20

        for kind in EXPECTED_LABEL:
            hits = 0
            for _ in range(n):
                base = dist.mean_ + rng.standard_normal(_N_DIMS) * dist.std_ * 0.3
                v, expected = perturb_vector(base, dist, kind, magnitude=3.5)  # type: ignore[arg-type]
                if expected in dist.classify(v):
                    hits += 1
            kind_recall = hits / n
            assert kind_recall >= 0.5, f"kind={kind} recall={kind_recall:.2f} < 0.5"