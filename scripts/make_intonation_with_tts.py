"""learner.wav + reference_text → TTS native 합성 후 prosody 분석.

native wav를 gTTS(Google Translate TTS, 무료·인터넷 필요)로 합성해 data/tts/에
캐시하고, make_intonation_from_wavs 파이프라인의 native 입력으로 넣는다.
원어민 녹음이 없을 때 reference 음성을 자동 생성하는 진입점.

Usage:
    python3 scripts/make_intonation_with_tts.py \\
        --learner <learner.wav> \\
        --text    "reference text" \\
        [--out data/output] [--tts-dir data/tts]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from make_intonation_from_wavs import run as run_intonation


def main(learner_wav: Path, text: str, out_dir: Path, tts_dir: Path) -> None:
    # gTTS는 torch 비의존이라 import 순서 제약 없음 (run_intonation 내부에서
    # Wav2Vec2 로딩 후 parselmouth/dtaidistance 지연 import는 그대로 유지).
    from core.tts import generate_tts

    clean_text = "".join(c for c in text if c not in ".·,!?。")
    print("TTS native 생성 중...")
    native_wav = generate_tts(text, cache_dir=tts_dir, filename=f"{clean_text}.wav")
    print(f"native TTS → {native_wav}")

    run_intonation(learner_wav=learner_wav, native_wav=native_wav, text=text, out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learner", required=True, help="학습자 WAV 경로")
    parser.add_argument("--text", required=True, help="reference 텍스트")
    parser.add_argument("--out", default="data/output", help="출력 디렉토리")
    parser.add_argument("--tts-dir", default="data/tts", help="TTS 캐시 디렉토리")
    args = parser.parse_args()
    main(
        learner_wav=Path(args.learner),
        text=args.text,
        out_dir=Path(args.out),
        tts_dir=Path(args.tts_dir),
    )
