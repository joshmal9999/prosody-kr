"""learner.wav + native.wav + reference_text → prosody JSON + plot 생성.

Usage:
    python3 scripts/make_intonation_json.py \\
        --learner <learner.wav> \\
        --native  <native.wav> \\
        --text    "reference text" \\
        [--out    artifacts/my_session]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.comparator import IntonationComparator
from core.f0_extractor import extract_f0
from core.metrics import compute_metrics
from core.plotter import ComparisonPlotter
from core.syllable_utils import segments_to_syllable_boundaries
from src.audio_to_ipa import AudioToIPARecognizer
from src.forced_alignment import force_align_candidate
from src.korean_ipa import pronunciation_to_ipa
from src.recognition import recognize_audio
from src.types import PronunciationCandidate


def _forced_align(recognizer: AudioToIPARecognizer, wav_path: Path, text: str):
    ipa_seq = pronunciation_to_ipa(text)
    candidate = PronunciationCandidate(pronunciation=text, ipa=ipa_seq, is_primary=True)
    recog = recognize_audio(recognizer, wav_path)
    vocab = recognizer.processor.tokenizer.get_vocab()
    blank_id = vocab[recognizer.processor.tokenizer.pad_token]
    return force_align_candidate(
        candidate,
        recog.logits,
        recog.frame_timestamps,
        label_to_id=vocab,
        blank_id=blank_id,
    )


def run(
    learner_wav: Path,
    native_wav: Path,
    text: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("모델 로딩 중...")
    recognizer = AudioToIPARecognizer()

    print("learner forced alignment 중...")
    learner_fa = _forced_align(recognizer, learner_wav, text)

    print("native forced alignment 중...")
    native_fa = _forced_align(recognizer, native_wav, text)

    # ── prosody JSON 저장 ────────────────────────────────────────────────────
    payload = {
        "reference_text": text,
        "native": {
            "wav": str(native_wav.resolve()),
            "phoneme_segments": [asdict(seg) for seg in native_fa.segments],
        },
        "learner": {
            "wav": str(learner_wav.resolve()),
            "phoneme_segments": [asdict(seg) for seg in learner_fa.segments],
        },
    }
    json_path = out_dir / "prosody.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON 저장 → {json_path}")

    # ── F0 추출 + 음절 경계 변환 ─────────────────────────────────────────────
    native_f0 = extract_f0(native_wav)
    learner_f0 = extract_f0(learner_wav)
    native_b = segments_to_syllable_boundaries(payload["native"]["phoneme_segments"])
    learner_b = segments_to_syllable_boundaries(payload["learner"]["phoneme_segments"])

    # ── segmental alignment + 메트릭 + plot ──────────────────────────────────
    comparisons = IntonationComparator().compare(
        native_f0, learner_f0,
        native_boundaries=native_b,
        learner_boundaries=learner_b,
    )
    metrics = compute_metrics(comparisons)

    syllable_labels = [c for c in text if c.strip() and c not in ".·,!?。"]
    fig = ComparisonPlotter(threshold=1).plot(
        comparisons, metrics,
        title=text,
        syllable_labels=syllable_labels,
    )
    plot_path = out_dir / "plot.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"plot 저장 → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", required=True, help="학습자 WAV 경로")
    parser.add_argument("--native",  required=True, help="원어민/TTS WAV 경로")
    parser.add_argument("--text",    required=True, help="reference 텍스트")
    parser.add_argument("--out",     default="artifacts/intonation_analysis", help="출력 디렉토리")
    args = parser.parse_args()

    run(
        learner_wav=Path(args.learner),
        native_wav=Path(args.native),
        text=args.text,
        out_dir=Path(args.out),
    )