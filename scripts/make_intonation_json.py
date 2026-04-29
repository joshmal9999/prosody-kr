"""native/learner wav 모두에 대해 forced alignment를 실행하고 prosody_input JSON을 생성한다.

Usage:
    python3 scripts/make_intonation_json.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_to_ipa import AudioToIPARecognizer
from src.forced_alignment import force_align_candidate
from src.korean_ipa import pronunciation_to_ipa
from src.recognition import recognize_audio
from src.types import PronunciationCandidate

TEXT = "시간이 멈춘 것 같았습니다"
NATIVE_WAV = Path("data/intonation_01_correct.wav")
LEARNER_WAV = Path("data/intonation_01_error.wav")
OUT_JSON = Path("artifacts/intonation_01/intonation_01.json")


def _run_forced_alignment(
    recognizer: AudioToIPARecognizer,
    wav_path: Path,
    text: str,
):
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


def main() -> None:
    print("모델 로딩 중...")
    recognizer = AudioToIPARecognizer()

    print(f"native 정렬 중: {NATIVE_WAV}")
    native_fa = _run_forced_alignment(recognizer, NATIVE_WAV, TEXT)

    print(f"learner 정렬 중: {LEARNER_WAV}")
    learner_fa = _run_forced_alignment(recognizer, LEARNER_WAV, TEXT)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_text": TEXT,
        "native": {
            "phoneme_segments": [asdict(seg) for seg in native_fa.segments],
        },
        "learner": {
            "phoneme_segments": [asdict(seg) for seg in learner_fa.segments],
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"저장 완료 → {OUT_JSON}  (native {len(native_fa.segments)}seg / learner {len(learner_fa.segments)}seg)")


if __name__ == "__main__":
    main()