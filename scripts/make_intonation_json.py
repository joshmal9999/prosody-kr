"""artifact JSON을 기반으로 native(TTS) forced alignment를 실행하고
prosody 비교용 JSON을 생성한다.

Usage:
    python3 scripts/make_intonation_json.py [artifact_json_path]

    artifact_json_path 미지정 시 DEFAULT_ARTIFACT 사용.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tts import generate_tts
from src.audio_to_ipa import AudioToIPARecognizer
from src.forced_alignment import force_align_candidate
from src.korean_ipa import pronunciation_to_ipa
from src.recognition import recognize_audio
from src.types import PronunciationCandidate

TTS_CACHE_DIR = Path("artifacts/tts_cache")
DEFAULT_ARTIFACT = Path(
    "artifacts/20260421_220712_176144/20260421_220712_176144.json"
)


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


def main(artifact_json: Path = DEFAULT_ARTIFACT) -> None:
    with open(artifact_json, encoding="utf-8") as f:
        artifact = json.load(f)

    text = artifact["reference"]["text"]
    artifact_dir = artifact_json.parent
    learner_wav = artifact_dir / artifact["artifact_bundle"]["audio_file_name"]
    learner_segments = artifact["prosody_input"]["phoneme_segments"]

    print(f"reference text: {text!r}")
    print(f"learner wav: {learner_wav}")
    print(f"learner segments: {len(learner_segments)}개")

    print(f"\nTTS 생성 중...")
    native_wav = generate_tts(text, cache_dir=TTS_CACHE_DIR)
    print(f"native wav: {native_wav}")

    print("모델 로딩 중...")
    recognizer = AudioToIPARecognizer()

    print("native forced alignment 중 (TTS)...")
    native_fa = _run_forced_alignment(recognizer, native_wav, text)

    artifact_id = artifact["artifact_bundle"]["artifact_id"]
    out_json = artifact_dir / f"{artifact_id}_prosody.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "artifact_id": artifact_id,
        "reference_text": text,
        "native": {
            "wav": str(native_wav),
            "phoneme_segments": [asdict(seg) for seg in native_fa.segments],
        },
        "learner": {
            "wav": str(learner_wav),
            "phoneme_segments": learner_segments,
        },
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"\n저장 완료 → {out_json}"
        f"  (native {len(native_fa.segments)}seg / learner {len(learner_segments)}seg)"
    )


if __name__ == "__main__":
    artifact_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARTIFACT
    main(artifact_path)