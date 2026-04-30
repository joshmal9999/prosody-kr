"""억양(Prosody) 분석 파이프라인 단일 진입점.

의존성:
    core/   — F0 추출, segmental alignment, 메트릭 계산 (외부 의존 없음)
    src/    — AudioToIPARecognizer (Wav2Vec2), forced alignment, Korean IPA 변환

사용법:
    from analyze import analyze

    results = analyze("artifacts/20260421_220712_176144/20260421_220712_176144.json")
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from core.comparator import IntonationComparator
from core.f0_extractor import extract_f0
from core.metrics import compute_metrics, to_dict
from core.syllable_utils import segments_to_syllable_boundaries
from core.tts import generate_tts
from src.audio_to_ipa import AudioToIPARecognizer
from src.forced_alignment import force_align_candidate
from src.korean_ipa import pronunciation_to_ipa
from src.recognition import recognize_audio
from src.types import PronunciationCandidate

_TTS_CACHE_DIR = Path("artifacts/tts_cache")

# Wav2Vec2 모델은 로딩 비용이 크므로 프로세스 내 싱글톤으로 유지
_recognizer: AudioToIPARecognizer | None = None


def _get_recognizer() -> AudioToIPARecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = AudioToIPARecognizer()
    return _recognizer


def _forced_align(wav_path: Path, text: str):
    recognizer = _get_recognizer()
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


def analyze(artifact_json: Path | str) -> list[dict]:
    """teammate artifact JSON → 음절별 억양 분석 결과.

    Args:
        artifact_json: 음소분석 파이프라인이 생성한 artifact JSON 경로.
                       status.evaluation_status == "ready" 조건을 통과한
                       artifact 여야 합니다.

    Returns:
        음절별 dict 리스트. 음절 수 = min(native 음절 수, learner 음절 수).
        모든 값은 JSON 직렬화 가능 (NaN → None).

        각 dict 필드:
            syllable_idx        (int)          음절 인덱스 (0부터)
            native_f0           (list[float])  50프레임 z-score F0, 무성=0 (원어민 TTS)
            learner_f0          (list[float])  50프레임 z-score F0, 무성=0 (학습자)
            joint_voiced_mask   (list[bool])   두 화자 모두 유성인 프레임
            native_duration     (float)        원어민 음절 지속시간 (초)
            learner_duration    (float)        학습자 음절 지속시간 (초)
            rmse                (float|None)   F0 RMSE (z-score 단위); 유성 프레임 없으면 None
            pearson             (float|None)   억양 흐름 유사도 -1~1; 유성 프레임 < 5이면 None
            slope_diff          (float|None)   피치 변화율 차이 native-learner; 유성 프레임 < 3이면 None
            voiced_frame_count  (int)          유효 유성 프레임 수
            duration_ratio      (float|None)   learner/native 지속시간 비율; native=0이면 None
    """

    # ── pipeline 개요 ────────────────────────────────────────────────────────
    # 음소분석 파이프라인의 artifact JSON은 learner 정보만 담고 있어
    # 억양 비교를 위한 native 기준값이 없다.
    # 이를 보완하기 위해 reference text로 TTS를 합성하고,
    # learner에 적용한 것과 동일한 forced alignment를 native에도 수행하여
    # 양쪽의 음소 경계를 확보한다.
    # 이 때문에 위에 _get_recognizer, _forced_align 함수는 억양분석 모듈(core)이 아닌
    # 음소분석 모듈(src)에 의존한다. **주의**
    #
    # 음소 경계 → 음절 경계로 변환한 뒤,
    # z-score 정규화(화자 간 음역대 차이 제거) + segmental alignment(음절 인덱스 1:1 매핑)로
    # native와 learner를 동일한 기준 위에 놓고 음절 단위로 비교한다.
    #
    # 음절별 비교 지표 3가지:
    #   RMSE        — F0 절대 차이. 억양이 얼마나 크게 벗어났는지
    #   Pearson     — 억양 흐름 유사도(-1~1). 올라가고 내려가는 방향이 같은지
    #   slope_diff  — 피치 변화율 차이(native - learner). 상승/하강 강도 비교
    #
    # 시각화와 API 전달에 필요한 모든 feature를 JSON 직렬화 가능한
    # list[dict] 형태로 반환한다.


    # ── Step 1. artifact JSON 파싱 ──────────────────────────────────────────
    # 음소분석 파이프라인 출력물. learner wav 경로와 음소 강제 정렬 결과를 읽는다.
    artifact_json = Path(artifact_json)
    with open(artifact_json, encoding="utf-8") as f:
        artifact = json.load(f)

    text = artifact["reference"]["text"]
    artifact_dir = artifact_json.parent
    learner_wav = artifact_dir / artifact["artifact_bundle"]["audio_file_name"]
    learner_segments = artifact["prosody_input"]["phoneme_segments"]  # src/ forced alignment 결과

    # ── Step 2. native(TTS) 생성 + forced alignment ──────────────────────────
    # TTS로 원어민 기준 오디오 합성 → Wav2Vec2 CTC로 음소별 시간 경계 추출
    # TTS 결과는 (text, voice, speed) 해시 기반으로 캐시됨 (재호출 비용 없음)
    native_wav = generate_tts(text, cache_dir=_TTS_CACHE_DIR)
    native_fa = _forced_align(native_wav, text)
    native_segments = [asdict(seg) for seg in native_fa.segments]

    # ── Step 3. F0 추출 + z-score 정규화 ────────────────────────────────────
    # parselmouth(Praat)로 각 wav의 피치 곡선 추출
    # 화자 간 음역대 차이를 제거하기 위해 z-score 정규화 적용
    native_f0 = extract_f0(native_wav)
    learner_f0 = extract_f0(learner_wav)

    # ── Step 4. 음소 segments → 음절 경계 변환 ──────────────────────────────
    # 각 음절의 (t_start, t_end) 를 확보. nucleus(모음) 기준으로 경계를 묶는다.
    native_boundaries = segments_to_syllable_boundaries(native_segments)
    learner_boundaries = segments_to_syllable_boundaries(learner_segments)

    # ── Step 5. Segmental Alignment — 음절별 F0 비교 ─────────────────────────
    # 음절 인덱스 1:1 매핑. 각 음절을 50프레임으로 리샘플 후 비교.
    # zip 기준이므로 음절 수 불일치 시 min(native, learner) 개수로 맞춰진다.
    comparisons = IntonationComparator().compare(
        native_f0, learner_f0,
        native_boundaries=native_boundaries,
        learner_boundaries=learner_boundaries,
    )

    # ── Step 6. 멀티 메트릭 계산 + 직렬화 ───────────────────────────────────
    # 음절별 RMSE / Pearson / slope_diff / duration_ratio 산출
    # NaN → None 변환으로 JSON 직렬화 보장
    metrics = compute_metrics(comparisons)
    return to_dict(comparisons, metrics)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/20260421_220712_176144/20260421_220712_176144.json"
    results = analyze(path)
    print(json.dumps(results, indent=2, ensure_ascii=False))