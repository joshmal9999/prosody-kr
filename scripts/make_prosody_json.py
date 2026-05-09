#!/usr/bin/env python3
"""learner WAV + reference text → *_prosody.json 생성.

Usage:
    python make_prosody_json.py <learner_wav> "<reference_text>"
    python make_prosody_json.py <learner_wav> "<reference_text>" --out artifacts/my_session/my_session_prosody.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from core.tts import generate_tts
from pronunciation_backend_pipeline import evaluate_pronunciation_file, get_default_recognizer
from src.forced_alignment import force_align_candidate
from src.korean_ipa import pronunciation_to_ipa
from src.recognition import recognize_audio
from src.types import PronunciationCandidate

_TTS_CACHE_DIR = Path("artifacts/tts_cache")


def _forced_align_wav(wav_path: Path, text: str, recognizer):
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


def make_prosody_json(
    learner_wav: str | Path,
    reference_text: str,
    *,
    out_path: str | Path | None = None,
) -> Path:
    """learner WAV + reference text 로부터 *_prosody.json 생성.

    Args:
        learner_wav:     학습자 WAV 파일 경로
        reference_text:  기준 텍스트 (한국어)
        out_path:        저장 경로. None이면 artifacts/<id>/<id>_prosody.json

    Returns:
        저장된 JSON 파일 경로
    """
    learner_wav = Path(learner_wav)
    rec = get_default_recognizer()

    # ── Step 1. learner 평가 파이프라인 (음소 segments + WAV 아티팩트 저장) ──
    result = evaluate_pronunciation_file(
        learner_wav, reference_text, recognizer=rec, save_artifacts=True
    )
    learner_segments = result["prosody_input"]["phoneme_segments"]
    learner_audio = result["prosody_input"]["audio_file_path"]
    artifact_id = Path(result["artifact_paths"]["artifact_dir"]).name

    # ── Step 2. native TTS 합성 + forced alignment ───────────────────────────
    native_wav = generate_tts(reference_text, cache_dir=_TTS_CACHE_DIR)
    native_fa = _forced_align_wav(native_wav, reference_text, rec)
    native_segments = [asdict(seg) for seg in native_fa.segments]

    # ── Step 3. prosody JSON 조립 + 저장 ────────────────────────────────────
    payload = {
        "artifact_id": artifact_id,
        "reference_text": reference_text,
        "native": {
            "wav": str(native_wav),
            "phoneme_segments": native_segments,
        },
        "learner": {
            "wav": learner_audio,
            "phoneme_segments": learner_segments,
        },
    }

    if out_path is None:
        out_path = Path(result["artifact_paths"]["artifact_dir"]) / f"{artifact_id}_prosody.json"

    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return Path(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prosody JSON 생성")
    parser.add_argument("learner_wav", help="학습자 WAV 파일 경로")
    parser.add_argument("reference_text", help="기준 텍스트")
    parser.add_argument("--out", default=None, help="출력 JSON 경로 (기본: artifacts/<id>/<id>_prosody.json)")
    args = parser.parse_args()

    saved = make_prosody_json(args.learner_wav, args.reference_text, out_path=args.out)
    print(f"저장: {saved}")