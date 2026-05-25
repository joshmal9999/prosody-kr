"""5 rule trigger 로직 — lens 출력 → list[Record].

lens-rule paradigm의 실행 코어. lens 신호(어절 단위 metric)를 받아 threshold
비교 후 Record를 만든다. severity quantize → flat list → severity 내림차순까지
deterministic. LLM은 결과만 받아 NL 합성·톤 조절.

threshold는 config/thresholds.toml에서 로드 (hardcode 금지 — reactive 조정용).
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np

from core.f0_extractor import F0Result
from core.features import delta_f0
from core.record import Record, RuleLabel, Severity, sort_by_severity

_HANGUL_LO, _HANGUL_HI = 0xAC00, 0xD7A3
_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "thresholds.toml"


def load_thresholds(path: str | Path | None = None) -> dict[str, float]:
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "rb") as f:
        return tomllib.load(f)["thresholds"]


def _quantize(value: float, threshold: float) -> Severity:
    return "major" if abs(value) >= 2 * threshold else "minor"


def _slice(
    times: np.ndarray, arr: np.ndarray, voiced: np.ndarray, t0: float, t1: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = (times >= t0) & (times < t1)
    return arr[mask], voiced[mask]


def _mean_voiced(arr: np.ndarray, voiced: np.ndarray) -> float | None:
    if not voiced.any():
        return None
    return float(arr[voiced].mean())


def _eojeol_syllable_ranges(eojeol_text: str) -> list[tuple[int, int]]:
    """어절별 (음절 start_idx, end_idx exclusive). EojeolSegmenter._group과 동일 매핑."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for w in eojeol_text.split():
        n = sum(1 for ch in w if _HANGUL_LO <= ord(ch) <= _HANGUL_HI)
        ranges.append((cursor, cursor + n))
        cursor += n
    return ranges


def _position_hint(pos: int, total: int, label: str) -> str:
    if total == 1:
        prefix = "유일 음절"
    elif pos == 0:
        prefix = "첫 음절"
    elif pos == total - 1:
        prefix = "마지막 음절"
    else:
        prefix = f"{pos + 1}번째 음절"
    return f"{prefix} ({label})" if label else prefix


def evaluate(
    native_f0: F0Result,
    learner_f0: F0Result,
    eojeol_native_spans: list[tuple[float, float]],
    eojeol_learner_spans: list[tuple[float, float]],
    eojeol_labels: list[str],
    syllable_native_spans: list[tuple[float, float]],
    syllable_learner_spans: list[tuple[float, float]],
    syllable_labels: list[str],
    eojeol_text: str,
    thresholds: dict[str, float] | None = None,
) -> list[Record]:
    """모든 어절에 대해 5 rule 평가 → flat list[Record], severity 내림차순."""
    th = thresholds or load_thresholds()
    records: list[Record] = []

    # 전체 발화에서 delta-f0 1회 계산 (어절 경계 미분 artifact 회피)
    n_delta = delta_f0(native_f0)
    l_delta = delta_f0(learner_f0)
    syll_ranges = _eojeol_syllable_ranges(eojeol_text)

    n_eojeols = min(len(eojeol_native_spans), len(eojeol_learner_spans))
    for idx in range(n_eojeols):
        n_t0, n_t1 = eojeol_native_spans[idx]
        l_t0, l_t1 = eojeol_learner_spans[idx]
        label = eojeol_labels[idx] if idx < len(eojeol_labels) else ""

        rec = _rule_pitch_shape(
            idx, label, native_f0, learner_f0,
            n_delta, l_delta, n_t0, n_t1, l_t0, l_t1, th,
        )
        if rec:
            records.append(rec)

        rec = _rule_pitch_offset(
            idx, label, native_f0, learner_f0,
            n_t0, n_t1, l_t0, l_t1, th["pitch_offset"],
        )
        if rec:
            records.append(rec)

        rec = _rule_eojeol_slow(
            idx, label, n_t0, n_t1, l_t0, l_t1, th["eojeol_slow"],
        )
        if rec:
            records.append(rec)

        if idx < len(syll_ranges):
            s0, s1 = syll_ranges[idx]
            rec = _rule_syllable_elongation(
                idx, label, s0, s1,
                syllable_native_spans, syllable_learner_spans, syllable_labels,
                th["syllable_elongation"],
            )
            if rec:
                records.append(rec)

    return sort_by_severity(records)


def _rule_pitch_shape(
    idx: int, label: str,
    native_f0: F0Result, learner_f0: F0Result,
    n_delta: np.ndarray, l_delta: np.ndarray,
    n_t0: float, n_t1: float, l_t0: float, l_t1: float,
    th: dict[str, float],
) -> Record | None:
    n_d, n_v = _slice(native_f0.times, n_delta, native_f0.voiced_mask, n_t0, n_t1)
    l_d, l_v = _slice(learner_f0.times, l_delta, learner_f0.voiced_mask, l_t0, l_t1)
    n_mean = _mean_voiced(n_d, n_v)
    l_mean = _mean_voiced(l_d, l_v)
    if n_mean is None or l_mean is None:
        return None
    diff = l_mean - n_mean
    th_r = th["pitch_rising_excess"]
    th_f = th["pitch_falling_excess"]
    if diff > th_r:
        rule: RuleLabel = "pitch_rising_excess"
        sev = _quantize(diff, th_r)
    elif diff < -th_f:
        rule = "pitch_falling_excess"
        sev = _quantize(diff, th_f)
    else:
        return None
    return Record(
        eojeol_idx=idx,
        rule_label=rule,
        severity=sev,
        trigger_lens="eojeol_dtw_delta",
        evidence_metrics={
            "eojeol_label": label,
            "learner_mean_delta": round(l_mean, 4),
            "native_mean_delta": round(n_mean, 4),
            "delta_diff": round(diff, 4),
        },
    )


def _rule_pitch_offset(
    idx: int, label: str,
    native_f0: F0Result, learner_f0: F0Result,
    n_t0: float, n_t1: float, l_t0: float, l_t1: float,
    threshold: float,
) -> Record | None:
    n_f, n_v = _slice(native_f0.times, native_f0.f0, native_f0.voiced_mask, n_t0, n_t1)
    l_f, l_v = _slice(learner_f0.times, learner_f0.f0, learner_f0.voiced_mask, l_t0, l_t1)
    n_mean = _mean_voiced(n_f, n_v)
    l_mean = _mean_voiced(l_f, l_v)
    if n_mean is None or l_mean is None:
        return None
    diff = l_mean - n_mean
    if abs(diff) < threshold:
        return None
    return Record(
        eojeol_idx=idx,
        rule_label="pitch_offset",
        severity=_quantize(diff, threshold),
        trigger_lens="f0_extractor",
        evidence_metrics={
            "eojeol_label": label,
            "learner_eojeol_z_mean": round(l_mean, 4),
            "native_eojeol_z_mean": round(n_mean, 4),
            "z_diff": round(diff, 4),
        },
    )


def _rule_eojeol_slow(
    idx: int, label: str,
    n_t0: float, n_t1: float, l_t0: float, l_t1: float,
    threshold: float,
) -> Record | None:
    n_dur = n_t1 - n_t0
    l_dur = l_t1 - l_t0
    if n_dur <= 0 or l_dur <= 0:
        return None
    ratio = l_dur / n_dur
    # 1.0이 정상 — 이탈 크기 (ratio - 1.0) 기준 trigger
    deviation = ratio - 1.0
    trigger = threshold - 1.0  # 예: threshold=1.4 → 이탈 ≥0.4면 trigger
    if abs(deviation) < trigger:
        return None
    return Record(
        eojeol_idx=idx,
        rule_label="eojeol_slow",
        severity=_quantize(deviation, trigger),
        trigger_lens="eojeol_noalign",
        evidence_metrics={
            "eojeol_label": label,
            "learner_eojeol_dur_sec": round(l_dur, 3),
            "native_eojeol_dur_sec": round(n_dur, 3),
            "duration_ratio": round(ratio, 3),
        },
    )


def _rule_syllable_elongation(
    idx: int, label: str,
    s0: int, s1: int,
    syll_native_spans: list[tuple[float, float]],
    syll_learner_spans: list[tuple[float, float]],
    syll_labels: list[str],
    threshold: float,
) -> Record | None:
    s1_eff = min(s1, len(syll_native_spans), len(syll_learner_spans))
    if s0 >= s1_eff:
        return None
    max_ratio = 0.0
    max_pos = -1
    learner_dur = native_dur = 0.0
    for i in range(s0, s1_eff):
        n_dur = syll_native_spans[i][1] - syll_native_spans[i][0]
        l_dur = syll_learner_spans[i][1] - syll_learner_spans[i][0]
        if n_dur <= 0:
            continue
        ratio = l_dur / n_dur
        if ratio > max_ratio:
            max_ratio, max_pos = ratio, i - s0
            learner_dur, native_dur = l_dur, n_dur
    if max_pos < 0 or max_ratio < threshold:
        return None
    syll_label = syll_labels[s0 + max_pos] if (s0 + max_pos) < len(syll_labels) else ""
    deviation = max_ratio - 1.0
    trigger = threshold - 1.0
    return Record(
        eojeol_idx=idx,
        rule_label="syllable_elongation",
        severity=_quantize(deviation, trigger),
        trigger_lens="syllable_noalign",
        syllable_hint=_position_hint(max_pos, s1_eff - s0, syll_label),
        evidence_metrics={
            "eojeol_label": label,
            "syllable_idx_in_eojeol": max_pos,
            "syllable_label": syll_label,
            "learner_duration_sec": round(learner_dur, 3),
            "native_duration_sec": round(native_dur, 3),
            "duration_ratio": round(max_ratio, 3),
        },
    )
