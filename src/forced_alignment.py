from __future__ import annotations

"""Frame-level CTC forced alignment for reference IPA tokens."""

from math import inf

import numpy as np

from src.label_to_ipa import ipa_tokens_to_labels
from src.types import AlignmentSegment, ForcedAlignmentResult, PronunciationCandidate


MAX_BRIDGED_BLANK_FRAMES = 20
MAX_EDGE_PADDING_FRAMES = 2


def _build_extended_sequence(target_ids: list[int], blank_id: int) -> list[int]:
    sequence = [blank_id]
    for token_id in target_ids:
        sequence.append(token_id)
        sequence.append(blank_id)
    return sequence


def _viterbi_ctc_path(log_probs: np.ndarray, target_ids: list[int], blank_id: int) -> tuple[list[int], float]:
    extended = _build_extended_sequence(target_ids, blank_id)
    num_frames = log_probs.shape[0]
    num_states = len(extended)

    dp = np.full((num_frames, num_states), -inf, dtype=np.float64)
    back = np.full((num_frames, num_states), -1, dtype=np.int32)

    dp[0, 0] = float(log_probs[0, blank_id])
    if num_states > 1:
        dp[0, 1] = float(log_probs[0, extended[1]])

    for frame in range(1, num_frames):
        for state in range(num_states):
            candidates = [(dp[frame - 1, state], state)]
            if state - 1 >= 0:
                candidates.append((dp[frame - 1, state - 1], state - 1))
            if (
                state - 2 >= 0
                and extended[state] != blank_id
                and extended[state] != extended[state - 2]
            ):
                candidates.append((dp[frame - 1, state - 2], state - 2))

            best_score, best_prev = max(candidates, key=lambda item: item[0])
            dp[frame, state] = best_score + float(log_probs[frame, extended[state]])
            back[frame, state] = best_prev

    end_candidates = [(dp[num_frames - 1, num_states - 1], num_states - 1)]
    if num_states > 1:
        end_candidates.append((dp[num_frames - 1, num_states - 2], num_states - 2))
    best_score, best_state = max(end_candidates, key=lambda item: item[0])

    states = [best_state]
    state = best_state
    for frame in range(num_frames - 1, 0, -1):
        state = int(back[frame, state])
        states.append(state)
    states.reverse()
    return states, float(best_score)


def _frame_boundaries(frame_timestamps: list[float], num_frames: int) -> list[float]:
    if num_frames <= 0:
        return [0.0]

    if len(frame_timestamps) >= num_frames + 1:
        return [float(timestamp) for timestamp in frame_timestamps[:num_frames + 1]]

    if len(frame_timestamps) == num_frames and num_frames > 1:
        audio_duration = float(frame_timestamps[-1])
        if audio_duration > 0.0:
            return [audio_duration * index / num_frames for index in range(num_frames + 1)]

    if frame_timestamps:
        frame_width = (
            float(frame_timestamps[1] - frame_timestamps[0])
            if len(frame_timestamps) > 1
            else 0.02
        )
        boundaries = [float(timestamp) for timestamp in frame_timestamps]
        while len(boundaries) < num_frames + 1:
            boundaries.append(boundaries[-1] + frame_width)
        return boundaries

    return [float(index) for index in range(num_frames + 1)]


def _segment_boundaries(label_frame_buckets: list[list[int]], num_frames: int) -> list[tuple[int, int] | None]:
    evidence_ranges: list[tuple[int, int] | None] = [
        (frame_indices[0], frame_indices[-1])
        if frame_indices else None
        for frame_indices in label_frame_buckets
    ]
    present = [index for index, frame_range in enumerate(evidence_ranges) if frame_range is not None]
    boundaries: list[tuple[int, int] | None] = [None for _ in label_frame_buckets]

    for label_index in present:
        frame_range = evidence_ranges[label_index]
        assert frame_range is not None
        raw_start, raw_end = frame_range
        start_boundary = max(0, raw_start - MAX_EDGE_PADDING_FRAMES)
        end_boundary = min(num_frames, raw_end + 1 + MAX_EDGE_PADDING_FRAMES)
        boundaries[label_index] = (start_boundary, max(start_boundary + 1, end_boundary))

    for previous_index, next_index in zip(present, present[1:]):
        previous_range = evidence_ranges[previous_index]
        next_range = evidence_ranges[next_index]
        previous_boundary = boundaries[previous_index]
        next_boundary = boundaries[next_index]
        assert previous_range is not None and next_range is not None
        assert previous_boundary is not None and next_boundary is not None

        previous_end_exclusive = previous_range[1] + 1
        next_start = next_range[0]
        blank_gap = next_start - previous_end_exclusive
        if blank_gap <= MAX_BRIDGED_BLANK_FRAMES:
            split = previous_end_exclusive + max(0, blank_gap) // 2
            boundaries[previous_index] = (previous_boundary[0], max(previous_boundary[0] + 1, split))
            boundaries[next_index] = (min(split, next_boundary[1] - 1), next_boundary[1])
        else:
            previous_end = min(previous_boundary[1], previous_end_exclusive + MAX_EDGE_PADDING_FRAMES)
            next_start_boundary = max(next_boundary[0], next_start - MAX_EDGE_PADDING_FRAMES)
            boundaries[previous_index] = (previous_boundary[0], max(previous_boundary[0] + 1, previous_end))
            boundaries[next_index] = (min(next_start_boundary, next_boundary[1] - 1), next_boundary[1])

    return boundaries


def force_align_candidate(
    candidate: PronunciationCandidate,
    logits: list[list[float]],
    frame_timestamps: list[float],
    label_to_id: dict[str, int],
    blank_id: int,
) -> ForcedAlignmentResult:
    if not logits:
        raise ValueError("Forced alignment requires frame-level logits.")

    token_symbols = [token.symbol for token in candidate.ipa.tokens]
    labels = ipa_tokens_to_labels(candidate.ipa.tokens)
    target_ids = [label_to_id[label] for label in labels]

    log_probs = np.asarray(logits, dtype=np.float64)
    log_probs = log_probs - np.logaddexp.reduce(log_probs, axis=1, keepdims=True)

    states, best_score = _viterbi_ctc_path(log_probs, target_ids, blank_id)
    num_frames = len(states)
    frame_boundaries = _frame_boundaries(frame_timestamps, num_frames)

    label_frame_buckets: list[list[int]] = [[] for _ in labels]
    blank_frames = 0
    extended = _build_extended_sequence(target_ids, blank_id)
    for frame_index, state in enumerate(states):
        symbol_id = extended[state]
        if symbol_id == blank_id:
            blank_frames += 1
            continue
        label_index = (state - 1) // 2
        label_frame_buckets[label_index].append(frame_index)

    segments: list[AlignmentSegment] = []
    confidences: list[float] = []
    segment_boundaries = _segment_boundaries(label_frame_buckets, num_frames)
    for index, frame_indices in enumerate(label_frame_buckets):
        if not frame_indices:
            continue

        boundary_range = segment_boundaries[index]
        if boundary_range is None:
            continue
        frame_start, frame_end_exclusive = boundary_range
        time_start = frame_boundaries[frame_start]
        time_end = frame_boundaries[min(frame_end_exclusive, len(frame_boundaries) - 1)]

        label_id = target_ids[index]
        probs = np.exp(log_probs[frame_indices, label_id])
        confidence = float(np.mean(probs))
        confidences.append(confidence)
        segments.append(
            AlignmentSegment(
                token=token_symbols[index],
                label=labels[index],
                start_time=time_start,
                end_time=time_end,
                frame_start=frame_start,
                frame_end=max(frame_start, frame_end_exclusive - 1),
                confidence=confidence,
            )
        )

    coverage = len(segments) / len(labels) if labels else 0.0
    avg_token_confidence = float(np.mean(confidences)) if confidences else 0.0
    blank_ratio = blank_frames / num_frames if num_frames else 1.0
    normalized_log_prob = best_score / max(num_frames, 1)

    return ForcedAlignmentResult(
        labels=labels,
        token_symbols=token_symbols,
        segments=segments,
        total_log_prob=best_score,
        normalized_log_prob=normalized_log_prob,
        avg_token_confidence=avg_token_confidence,
        coverage=coverage,
        blank_ratio=blank_ratio,
    )