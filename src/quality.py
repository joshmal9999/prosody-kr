from __future__ import annotations

"""Audio quality and forced-alignment confidence gates."""

import math

import numpy as np

from src.types import AlignmentConfidenceReport, AudioQualityReport, ForcedAlignmentResult


def analyze_audio_quality(audio: np.ndarray, sampling_rate: int) -> AudioQualityReport:
    duration_sec = len(audio) / float(sampling_rate) if sampling_rate else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-8))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.995)) if audio.size else 1.0

    frame_size = max(1, int(sampling_rate * 0.02))
    frames = [audio[index:index + frame_size] for index in range(0, len(audio), frame_size)] or [audio]
    frame_energy = [float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0 for frame in frames]
    silence_ratio = float(sum(1 for energy in frame_energy if energy < 0.02) / len(frame_energy)) if frame_energy else 1.0

    reasons: list[str] = []
    if duration_sec < 0.6:
        reasons.append("음성이 너무 짧습니다.")
    if rms_db < -35.0:
        reasons.append("입력 음량이 너무 낮습니다.")
    if silence_ratio > 0.85:
        reasons.append("무음 비율이 너무 높습니다.")
    if clipping_ratio > 0.08:
        reasons.append("입력 신호에 clipping이 많습니다.")

    return AudioQualityReport(
        passed=not reasons,
        duration_sec=duration_sec,
        rms_db=rms_db,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        reasons=reasons,
    )


def assess_alignment_confidence(result: ForcedAlignmentResult) -> AlignmentConfidenceReport:
    """
    Check whether frame-level forced alignment is reliable enough.

    Policy:
    - Do not fail only because one token has extremely low confidence.
    - Use ratios instead of a single minimum value.
    - Keep thresholds realistic for CTC forced alignment.
    """

    reasons: list[str] = []

    confidences = [segment.confidence for segment in result.segments]
    token_count = max(1, len(confidences))

    min_token_confidence = min(confidences) if confidences else 0.0
    low_conf_count = sum(conf < 0.20 for conf in confidences)
    very_low_conf_count = sum(conf < 0.05 for conf in confidences)

    low_conf_ratio = low_conf_count / token_count
    very_low_conf_ratio = very_low_conf_count / token_count

    debug_notes: list[str] = []
    durations = [max(0.0, segment.end_time - segment.start_time) for segment in result.segments]
    gaps = [
        max(0.0, result.segments[index].start_time - result.segments[index - 1].end_time)
        for index in range(1, len(result.segments))
    ]
    median_duration = float(np.median(durations)) if durations else 0.0
    max_duration = max(durations) if durations else 0.0
    max_gap = max(gaps) if gaps else 0.0
    max_tail_gap = max(gaps[-2:]) if len(gaps) >= 2 else (gaps[-1] if gaps else 0.0)
    final_confidences = confidences[-2:]

    if result.coverage < 0.90:
        reasons.append(
            f"정답 음소 대부분이 시간축에 안정적으로 배치되지 않았습니다. "
            f"coverage={result.coverage:.3f}"
        )

    if result.avg_token_confidence < 0.45:
        reasons.append(
            f"정렬 경로의 평균 음소 신뢰도가 낮습니다. "
            f"avg_token_confidence={result.avg_token_confidence:.3f}"
        )

    if low_conf_ratio > 0.45:
        reasons.append(
            f"신뢰도 0.20 미만 음소 비율이 높습니다. "
            f"low_conf_ratio={low_conf_ratio:.3f}"
        )

    if very_low_conf_ratio > 0.30:
        reasons.append(
            f"신뢰도 0.05 미만 음소 비율이 높습니다. "
            f"very_low_conf_ratio={very_low_conf_ratio:.3f}"
        )

    # Warning only, not a hard failure.
    if len(confidences) >= 10 and min_token_confidence < 0.005:
        debug_notes.append(
            f"min_token_confidence={min_token_confidence:.6f}"
        )

    if result.normalized_log_prob < -1.50:
        reasons.append(
            f"전체 정렬 경로의 로그확률이 낮습니다. "
            f"normalized_log_prob={result.normalized_log_prob:.3f}"
        )

    if result.blank_ratio > 0.97:
        reasons.append(
            f"blank frame 비율이 너무 높습니다. "
            f"blank_ratio={result.blank_ratio:.3f}"
        )

    if len(final_confidences) == 2 and all(conf < 0.05 for conf in final_confidences):
        reasons.append(
            "Forced alignment confidence is too low on the final phones. "
            f"final_confidences={[round(conf, 6) for conf in final_confidences]}"
        )

    if median_duration > 0 and max_duration > max(0.75, median_duration * 8.0):
        reasons.append(
            "Forced alignment timing has an abnormally long phone interval. "
            f"max_duration={max_duration:.3f}, median_duration={median_duration:.3f}"
        )

    if max_tail_gap > 0.30:
        reasons.append(
            "Forced alignment timing has an abnormal gap near the final phones. "
            f"max_tail_gap={max_tail_gap:.3f}"
        )
    elif max_gap > 1.20:
        reasons.append(
            "Forced alignment timing has an abnormal internal phone gap. "
            f"max_gap={max_gap:.3f}"
        )

    if reasons:
        message = " ".join(reasons)
    else:
        message = "forced alignment를 신뢰할 수 있습니다."

    if debug_notes:
        message = message + " debug: " + ", ".join(debug_notes)

    return AlignmentConfidenceReport(
        passed=not reasons,
        avg_token_confidence=result.avg_token_confidence,
        coverage=result.coverage,
        normalized_log_prob=result.normalized_log_prob,
        message=message,
    )