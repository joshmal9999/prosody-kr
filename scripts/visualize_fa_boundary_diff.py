"""Old FA vs New FA 음절 경계 시각화.

Usage:
    python3 scripts/visualize_fa_boundary_diff.py \
        --wav1 data/윤재욱.wav \
        --wav2 data/최민호.wav \
        --text "안녕하세요 저는 오상영입니다" \
        --out artifacts/fa_diff.html
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from math import inf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.f0_extractor import extract_f0
from core.syllable_utils import segments_to_syllable_boundaries
from src.audio_to_ipa import AudioToIPARecognizer
from src.forced_alignment import (
    MAX_BRIDGED_BLANK_FRAMES,
    MAX_EDGE_PADDING_FRAMES,
    _build_extended_sequence,
    _frame_boundaries,
    _segment_boundaries,
    _viterbi_ctc_path,
)
from src.korean_ipa import pronunciation_to_ipa
from src.label_to_ipa import ipa_tokens_to_labels
from src.recognition import recognize_audio
from src.types import PronunciationCandidate


# ── old FA: 원본 Viterbi frame만 사용, 패딩 없음 ──────────────────────────────

def _old_segment_times(
    label_frame_buckets: list[list[int]],
    frame_timestamps: list[float],
) -> list[tuple[float, float] | None]:
    results: list[tuple[float, float] | None] = []
    for frame_indices in label_frame_buckets:
        if not frame_indices:
            results.append(None)
            continue
        frame_start = frame_indices[0]
        frame_end = frame_indices[-1]
        time_start = frame_timestamps[frame_start] if frame_timestamps else float(frame_start)
        if frame_timestamps and frame_end + 1 < len(frame_timestamps):
            time_end = frame_timestamps[frame_end + 1]
        elif frame_timestamps:
            fw = frame_timestamps[1] - frame_timestamps[0] if len(frame_timestamps) > 1 else 0.02
            time_end = frame_timestamps[frame_end] + fw
        else:
            time_end = float(frame_end + 1)
        results.append((time_start, time_end))
    return results


# ── 공통: recognition + Viterbi ──────────────────────────────────────────────

def _run_alignment(recognizer: AudioToIPARecognizer, wav_path: Path, clean_text: str):
    ipa_seq = pronunciation_to_ipa(clean_text)
    candidate = PronunciationCandidate(pronunciation=clean_text, ipa=ipa_seq, is_primary=True)
    recog = recognize_audio(recognizer, wav_path)

    vocab = recognizer.processor.tokenizer.get_vocab()
    blank_id = vocab[recognizer.processor.tokenizer.pad_token]
    labels = ipa_tokens_to_labels(candidate.ipa.tokens)
    target_ids = [vocab[label] for label in labels]

    log_probs = np.asarray(recog.logits, dtype=np.float64)
    log_probs = log_probs - np.logaddexp.reduce(log_probs, axis=1, keepdims=True)

    states, _ = _viterbi_ctc_path(log_probs, target_ids, blank_id)
    num_frames = len(states)
    extended = _build_extended_sequence(target_ids, blank_id)

    label_frame_buckets: list[list[int]] = [[] for _ in labels]
    for frame_idx, state in enumerate(states):
        sym = extended[state]
        if sym == blank_id:
            continue
        label_frame_buckets[(state - 1) // 2].append(frame_idx)

    frame_ts = recog.frame_timestamps  # N+1 boundaries (new recognition.py)

    # new FA boundaries
    new_fb = _frame_boundaries(frame_ts, num_frames)
    new_seg_b = _segment_boundaries(label_frame_buckets, num_frames)
    new_times: list[tuple[float, float] | None] = []
    for idx, fi in enumerate(label_frame_buckets):
        if not fi or new_seg_b[idx] is None:
            new_times.append(None)
            continue
        fs, fe = new_seg_b[idx]
        new_times.append((new_fb[fs], new_fb[min(fe, len(new_fb) - 1)]))

    # old FA boundaries (N center 방식 시뮬레이션: N+1 boundary → center로 변환)
    if len(frame_ts) == num_frames + 1:
        centers = [(frame_ts[i] + frame_ts[i + 1]) / 2 for i in range(num_frames)]
    else:
        centers = list(frame_ts[:num_frames])
    old_times = _old_segment_times(label_frame_buckets, centers)

    token_symbols = [t.symbol for t in candidate.ipa.tokens]
    positions = [t.syllable_position for t in candidate.ipa.tokens]

    def _to_segments(times_list):
        segs = []
        for idx, t in enumerate(times_list):
            if t is None:
                continue
            segs.append({"start_time": t[0], "end_time": t[1], "token": token_symbols[idx]})
        return segs

    old_segs = _to_segments(old_times)
    new_segs = _to_segments(new_times)

    # positions는 None 제외한 순서와 일치해야 함
    present_pos = [positions[i] for i, t in enumerate(old_times) if t is not None]

    old_boundaries = segments_to_syllable_boundaries(old_segs, present_pos) if len(old_segs) == len(present_pos) else []
    new_boundaries = segments_to_syllable_boundaries(new_segs, present_pos) if len(new_segs) == len(present_pos) else []

    return old_boundaries, new_boundaries, candidate.ipa.tokens


# ── 시각화 ────────────────────────────────────────────────────────────────────

def _boundary_shapes(boundaries, color, y0, y1, dash="solid"):
    shapes = []
    for i, (t_start, t_end) in enumerate(boundaries):
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=t_start, x1=t_end, y0=y0, y1=y1,
            fillcolor=color, opacity=0.12, line_width=0,
        ))
        shapes.append(dict(
            type="line", xref="x", yref="y",
            x0=t_start, x1=t_start, y0=y0, y1=y1,
            line=dict(color=color, width=1.5, dash=dash),
        ))
        shapes.append(dict(
            type="line", xref="x", yref="y",
            x0=t_end, x1=t_end, y0=y0, y1=y1,
            line=dict(color=color, width=1.5, dash=dash),
        ))
    return shapes


def run(wav1: Path, wav2: Path, text: str, out_path: Path) -> None:
    clean_text = "".join(c for c in text if c not in ".·,!?。")
    print("모델 로딩 중...")
    recognizer = AudioToIPARecognizer()

    print(f"[1] {wav1.name} alignment...")
    old_b1, new_b1, tokens = _run_alignment(recognizer, wav1, clean_text)
    f0_1 = extract_f0(wav1)

    print(f"[2] {wav2.name} alignment...")
    old_b2, new_b2, _ = _run_alignment(recognizer, wav2, clean_text)
    f0_2 = extract_f0(wav2)

    token_str = " ".join(t.symbol for t in tokens)
    print(f"tokens: {token_str}")
    print(f"{wav1.name}: old={len(old_b1)}음절  new={len(new_b1)}음절")
    print(f"{wav2.name}: old={len(old_b2)}음절  new={len(new_b2)}음절")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[wav1.name, wav2.name],
        shared_xaxes=False,
        vertical_spacing=0.12,
    )

    for row, (f0r, old_b, new_b, name) in enumerate(
        [(f0_1, old_b1, new_b1, wav1.name), (f0_2, old_b2, new_b2, wav2.name)],
        start=1,
    ):
        voiced_t = f0r.times[f0r.voiced_mask]
        voiced_f0 = f0r.f0_raw[f0r.voiced_mask]

        fig.add_trace(go.Scatter(
            x=voiced_t, y=voiced_f0,
            mode="lines", name=f"F0 ({name})",
            line=dict(color="#2196F3", width=1.8),
            showlegend=(row == 1),
        ), row=row, col=1)

        y_min = float(voiced_f0.min()) * 0.95 if len(voiced_f0) else 0
        y_max = float(voiced_f0.max()) * 1.05 if len(voiced_f0) else 1

        shapes_old = _boundary_shapes(old_b, "#E53935", y_min, y_max, dash="dot")
        shapes_new = _boundary_shapes(new_b, "#43A047", y_min, y_max, dash="solid")

        # plotly row offset for shapes
        yref = "y" if row == 1 else "y2"
        xref = "x" if row == 1 else "x2"
        for s in shapes_old + shapes_new:
            s["yref"] = yref
            s["xref"] = xref

        fig.layout.shapes = list(fig.layout.shapes or []) + shapes_old + shapes_new

        # 음절 레이블 (new FA 기준 중간점)
        for t_start, t_end in new_b:
            fig.add_annotation(
                x=(t_start + t_end) / 2, y=y_max,
                text="▲", showarrow=False,
                font=dict(color="#43A047", size=10),
                xref=xref, yref=yref,
                row=row, col=1,
            )

    # 범례용 더미 trace
    for label, color, dash in [
        ("old FA 경계 (red·dot)", "#E53935", "dot"),
        ("new FA 경계 (green·solid)", "#43A047", "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=label,
            line=dict(color=color, dash=dash, width=2),
        ))

    fig.update_layout(
        title=f"Old FA vs New FA 음절 경계 비교<br><sup>text: {text}</sup>",
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.08),
    )
    fig.update_yaxes(title_text="F0 (Hz)")
    fig.update_xaxes(title_text="time (s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"\nHTML 저장 → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav1", default="data/윤재욱.wav")
    parser.add_argument("--wav2", default="data/최민호.wav")
    parser.add_argument("--text", default="안녕하세요 저는 오상영입니다")
    parser.add_argument("--out",  default="artifacts/fa_diff.html")
    args = parser.parse_args()
    run(Path(args.wav1), Path(args.wav2), args.text, Path(args.out))
