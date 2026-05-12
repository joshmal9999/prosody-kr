from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from core.comparator import IntonationComparator, SyllableComparison
from core.f0_extractor import extract_f0
from core.metrics import SyllableMetrics, compute_metrics, to_dict
from core.syllable_utils import segments_to_syllable_boundaries
from src.korean_ipa import pronunciation_to_ipa

ARTIFACT_DIR = Path(__file__).parent.parent / "artifacts" / "20260421_220712_176144"


# ── syllable_utils ────────────────────────────────────────────────────────────

class TestSegmentsToSyllableBoundaries:
    def _seg(self, start, end):
        return {"start_time": start, "end_time": end}

    def test_cv_structure(self):
        # 조 = tɕ + o
        segs = [self._seg(0.0, 0.05), self._seg(0.05, 0.15)]
        positions = ["onset", "nucleus"]
        assert segments_to_syllable_boundaries(segs, positions) == [(0.0, 0.15)]

    def test_cvc_v_with_pause(self):
        # 각아 = k + a + k̚ ... (pause) ... + a
        # 받침 k̚는 각의 coda로 묶이고, 다음 음절 아 사이의 묵음은 어느 boundary에도 포함되지 않아야 함
        segs = [
            self._seg(0.00, 0.05),  # k    (onset)
            self._seg(0.05, 0.15),  # a    (nucleus)
            self._seg(0.15, 0.20),  # k̚   (coda)
            self._seg(0.30, 0.40),  # a    (nucleus) — 0.20~0.30 사이 묵음
        ]
        positions = ["onset", "nucleus", "coda", "nucleus"]
        boundaries = segments_to_syllable_boundaries(segs, positions)
        assert boundaries == [(0.0, 0.20), (0.30, 0.40)]

    def test_silent_onset_consecutive_nuclei(self):
        # 아이 = a + i (둘 다 ㅇ초성이라 onset token 없음)
        segs = [self._seg(0.0, 0.10), self._seg(0.10, 0.20)]
        positions = ["nucleus", "nucleus"]
        assert segments_to_syllable_boundaries(segs, positions) == [
            (0.0, 0.10), (0.10, 0.20),
        ]

    def test_length_mismatch_raises(self):
        segs = [self._seg(0.0, 0.05)]
        with pytest.raises(ValueError):
            segments_to_syllable_boundaries(segs, ["onset", "nucleus"])

    def test_no_nucleus_returns_empty(self):
        segs = [self._seg(0.0, 0.05), self._seg(0.05, 0.10)]
        assert segments_to_syllable_boundaries(segs, ["onset", "coda"]) == []


# ── compute_metrics ───────────────────────────────────────────────────────────

def _make_syllable(
    native_f0: np.ndarray,
    learner_f0: np.ndarray,
    voiced_mask: np.ndarray | None = None,
    idx: int = 0,
    native_dur: float = 0.3,
    learner_dur: float = 0.3,
) -> SyllableComparison:
    if voiced_mask is None:
        voiced_mask = np.ones(len(native_f0), dtype=bool)
    return SyllableComparison(
        syllable_idx=idx,
        native_f0=native_f0,
        learner_f0=learner_f0,
        joint_voiced_mask=voiced_mask,
        native_duration=native_dur,
        learner_duration=learner_dur,
    )


class TestComputeMetrics:
    def test_returns_one_result_per_syllable(self):
        comparisons = [_make_syllable(np.zeros(50), np.zeros(50), idx=i) for i in range(3)]
        assert len(compute_metrics(comparisons)) == 3

    def test_identical_signals_give_zero_rmse(self):
        f0 = np.random.default_rng(0).standard_normal(50).astype(np.float32)
        results = compute_metrics([_make_syllable(f0, f0)])
        assert results[0].rmse == pytest.approx(0.0)

    def test_all_unvoiced_gives_nan(self):
        voiced_none = np.zeros(50, dtype=bool)
        results = compute_metrics([_make_syllable(np.ones(50), np.zeros(50), voiced_none)])
        assert results[0].voiced_frame_count == 0
        assert math.isnan(results[0].rmse)
        assert math.isnan(results[0].pearson)
        assert math.isnan(results[0].slope_diff)

    def test_rmse_value_is_correct(self):
        results = compute_metrics([_make_syllable(np.zeros(50), np.ones(50))])
        assert results[0].rmse == pytest.approx(1.0)
        assert results[0].voiced_frame_count == 50

    def test_pearson_nan_when_too_few_voiced(self):
        voiced = np.zeros(50, dtype=bool)
        voiced[:3] = True
        f0 = np.random.default_rng(1).standard_normal(50)
        results = compute_metrics([_make_syllable(f0, f0 * 0.5, voiced)])
        assert math.isnan(results[0].pearson)

    def test_duration_ratio(self):
        results = compute_metrics([
            _make_syllable(np.zeros(50), np.zeros(50), native_dur=0.2, learner_dur=0.4)
        ])
        assert results[0].duration_ratio == pytest.approx(2.0)


# ── segmental alignment 통합 (real audio) ─────────────────────────────────────

INTONATION_JSON = ARTIFACT_DIR / "20260421_220712_176144_prosody.json"

_intonation_files_exist = INTONATION_JSON.exists()


@pytest.mark.skipif(
    not _intonation_files_exist,
    reason="prosody JSON 없음",
)
class TestIntonationComparison:
    def _load(self):
        with open(INTONATION_JSON) as f:
            data = json.load(f)
        native_f0 = extract_f0(data["native"]["wav"])
        learner_f0 = extract_f0(data["learner"]["wav"])
        clean_text = "".join(c for c in data["reference_text"] if c not in ".·,!?。")
        positions = [t.syllable_position for t in pronunciation_to_ipa(clean_text).tokens]
        native_b = segments_to_syllable_boundaries(data["native"]["phoneme_segments"], positions)
        learner_b = segments_to_syllable_boundaries(data["learner"]["phoneme_segments"], positions)
        return native_f0, learner_f0, native_b, learner_b

    def test_voiced_syllables_exist(self):
        native_f0, learner_f0, native_b, learner_b = self._load()
        comparisons = IntonationComparator().compare(
            native_f0, learner_f0,
            native_boundaries=native_b, learner_boundaries=learner_b,
        )
        metrics = compute_metrics(comparisons)
        assert len(metrics) == len(native_b)
        assert any(not math.isnan(m.rmse) for m in metrics)

    def test_same_audio_gives_zero_rmse(self):
        native_f0, _, native_b, _ = self._load()
        comparisons = IntonationComparator().compare(
            native_f0, native_f0,
            native_boundaries=native_b, learner_boundaries=native_b,
        )
        for m in compute_metrics(comparisons):
            if not math.isnan(m.rmse):
                assert m.rmse == pytest.approx(0.0, abs=1e-5)

    def test_to_dict_is_json_serializable(self):
        import json as json_mod
        native_f0, learner_f0, native_b, learner_b = self._load()
        comparisons = IntonationComparator().compare(
            native_f0, learner_f0,
            native_boundaries=native_b, learner_boundaries=learner_b,
        )
        metrics = compute_metrics(comparisons)
        result = to_dict(comparisons, metrics)
        json_mod.dumps(result)  # NaN이 있으면 여기서 TypeError 발생

    def test_plot_native_vs_learner(self):
        import matplotlib.pyplot as plt
        from core.plotter import ComparisonPlotter

        with open(INTONATION_JSON) as f:
            data = json.load(f)
        ref_text = data.get("reference_text", "")
        syllable_labels = [c for c in ref_text if c.strip() and c not in ".·,!?。"]

        native_f0, learner_f0, native_b, learner_b = self._load()
        comparisons = IntonationComparator().compare(
            native_f0, learner_f0,
            native_boundaries=native_b, learner_boundaries=learner_b,
        )
        metrics = compute_metrics(comparisons)
        fig = ComparisonPlotter(threshold=1).plot(
            comparisons, metrics,
            title=ref_text,
            syllable_labels=syllable_labels,
        )
        out = ARTIFACT_DIR / "plot.png"
        fig.savefig(out)
        assert fig is not None