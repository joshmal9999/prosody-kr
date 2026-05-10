"""한국인 화자 분포 빌드 스크립트.

TL10 라벨링 JSON + TS10 WAV → 문장별 어절 분포 JSON 저장.

Usage:
    python3 scripts/build_korean_distribution.py [--out artifacts/korean_distribution.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from core.distribution import GaussianEojeolDistribution
from core.eojeol_vector import extract_eojeol_vector
from core.f0_extractor import extract_f0
from core.syllable_utils import segments_to_syllable_boundaries
from src.audio_to_ipa import AudioToIPARecognizer
from src.forced_alignment import force_align_candidate
from src.korean_ipa import pronunciation_to_ipa
from src.recognition import recognize_audio
from src.types import PronunciationCandidate

LABEL_ROOT = Path("datasets/014.다화자 음성합성 데이터/01.데이터/1.Training/라벨링데이터/TL10/1.남성/4800문장")
WAV_ROOT   = Path("datasets/014.다화자 음성합성 데이터/01.데이터/1.Training/원천데이터/TS10/1.남성/4800문장")

MIN_SPEAKERS = 12
MIN_EOJEOLS = 3
MAX_EOJEOLS = 6
N_SENTENCES = 50


# ── 인덱스 빌드 ──────────────────────────────────────────────────────────────

def build_index() -> dict[str, list[str]]:
    print("TL10 JSON 스캔 중...")
    text_to_wavs: dict[str, list[str]] = defaultdict(list)
    for json_path in sorted(LABEL_ROOT.rglob("*.json")):
        with open(json_path, encoding="utf-8") as f:
            d = json.load(f)
        text = d["전사정보"]["OrgLabelText"].strip()
        speaker_id = json_path.parent.name
        wav_name = d["파일정보"]["FileName"]
        wav_path = WAV_ROOT / speaker_id / wav_name
        if wav_path.exists():
            text_to_wavs[text].append(str(wav_path))
    print(f"  고유 문장: {len(text_to_wavs):,}개")
    return dict(text_to_wavs)


def select_sentences(index: dict[str, list[str]]) -> list[str]:
    candidates = [
        (text, wavs) for text, wavs in index.items()
        if len(wavs) >= MIN_SPEAKERS
        and MIN_EOJEOLS <= len(text.split()) <= MAX_EOJEOLS
    ]
    candidates.sort(key=lambda x: -len(x[1]))
    selected = [text for text, _ in candidates[:N_SENTENCES]]
    print(f"  선정 문장: {len(selected)}개 (12명+, {MIN_EOJEOLS}~{MAX_EOJEOLS}어절)")
    return selected


# ── WAV 단위 처리 ────────────────────────────────────────────────────────────

def _ipa_token_counts(eojeols: list[str]) -> list[int]:
    return [len(pronunciation_to_ipa(ej).tokens) for ej in eojeols]


_N_CONTOUR_FRAMES = 50


def _resample_f0(f0_result, t_start: float, t_end: float) -> np.ndarray:
    """구간 F0를 50프레임으로 리샘플 (z-score 기준)."""
    mask = (f0_result.times >= t_start) & (f0_result.times < t_end)
    f0 = f0_result.f0[mask]
    if len(f0) == 0:
        return np.zeros(_N_CONTOUR_FRAMES)
    src = np.linspace(0, _N_CONTOUR_FRAMES - 1, len(f0))
    return np.interp(np.arange(_N_CONTOUR_FRAMES, dtype=float), src, f0)


def _compute_mean_contours(
    speaker_contours: list[list[np.ndarray]],
) -> list[list[float]]:
    """화자 × 음절 contour 리스트 → 음절별 frame-wise mean.

    음절 수가 다른 화자는 제외 (mode 기준).
    """
    if not speaker_contours:
        return []
    from collections import Counter
    syl_counts = Counter(len(sc) for sc in speaker_contours)
    canonical_n = syl_counts.most_common(1)[0][0]
    filtered = [sc for sc in speaker_contours if len(sc) == canonical_n]
    if not filtered:
        return []
    # shape: (n_speakers, n_syllables, 50) → mean over speakers
    stacked = np.array(filtered)          # (S, N_syl, 50)
    mean_contour = stacked.mean(axis=0)   # (N_syl, 50)
    return mean_contour.tolist()


def process_wav(
    recognizer: AudioToIPARecognizer,
    wav_path: Path,
    text: str,
    eojeols: list[str],
    token_counts: list[int],
) -> tuple[list[np.ndarray], list[list[np.ndarray]]] | None:
    """WAV 1개 → (어절별 12-dim vector, 어절별 음절 contour 리스트). 실패 시 None."""
    try:
        recog = recognize_audio(recognizer, wav_path)
        vocab = recognizer.processor.tokenizer.get_vocab()
        blank_id = vocab[recognizer.processor.tokenizer.pad_token]

        ipa_full = pronunciation_to_ipa(text)
        candidate = PronunciationCandidate(pronunciation=text, ipa=ipa_full, is_primary=True)
        fa = force_align_candidate(
            candidate, recog.logits, recog.frame_timestamps,
            label_to_id=vocab, blank_id=blank_id,
        )

        segs = [
            {"token": s.token, "start_time": s.start_time, "end_time": s.end_time}
            for s in fa.segments
        ]

        if len(segs) != sum(token_counts):
            return None

        f0_result = extract_f0(wav_path)

        vectors: list[np.ndarray] = []
        syl_contours: list[list[np.ndarray]] = []  # [eojeol][syllable] → (50,)
        offset = 0
        for ej, n in zip(eojeols, token_counts):
            ej_segs = segs[offset: offset + n]
            offset += n
            syl_b = segments_to_syllable_boundaries(ej_segs)
            t_start = ej_segs[0]["start_time"]
            t_end   = ej_segs[-1]["end_time"]
            vectors.append(extract_eojeol_vector(f0_result, (t_start, t_end), syl_b))
            syl_contours.append([_resample_f0(f0_result, sb, eb) for sb, eb in syl_b])

        return vectors, syl_contours

    except Exception:
        return None


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main(out_path: Path) -> None:
    t0 = time.time()

    index = build_index()
    sentences = select_sentences(index)

    print("\n모델 로딩...")
    recognizer = AudioToIPARecognizer()

    output: dict = {
        "created_at": datetime.now().isoformat(),
        "n_sentences": len(sentences),
        "sentences": {},
    }
    skipped: list[str] = []

    for s_idx, text in enumerate(sentences):
        eojeols = text.split()
        wav_paths = index[text]
        token_counts = _ipa_token_counts(eojeols)
        print(f"\n[{s_idx+1}/{len(sentences)}] {text!r}  ({len(wav_paths)}명)")

        eojeol_vecs: list[list[np.ndarray]] = [[] for _ in eojeols]
        # [eojeol_i][speaker_j][syllable_k] → (50,) contour
        eojeol_syl_contours: list[list[list[np.ndarray]]] = [[] for _ in eojeols]

        for wav_str in wav_paths:
            wav = Path(wav_str)
            result = process_wav(recognizer, wav, text, eojeols, token_counts)
            if result is None:
                skipped.append(f"{text} | {wav.name}")
                print(f"  ✗ {wav.name}")
                continue
            vecs, syl_contours = result
            for i, (v, sc) in enumerate(zip(vecs, syl_contours)):
                eojeol_vecs[i].append(v)
                eojeol_syl_contours[i].append(sc)
            print(f"  ✓ {wav.name}")

        eojeol_entries: list[dict] = []
        for i, (ej, vecs) in enumerate(zip(eojeols, eojeol_vecs)):
            if len(vecs) < 2:
                print(f"  [!] {ej!r}: sample {len(vecs)}개 — 분포 생략")
                continue
            dist = GaussianEojeolDistribution()
            dist.fit(np.array(vecs))

            # 음절 평균 contour: 화자 수가 같은 음절만 frame-wise mean
            mean_syl_contours = _compute_mean_contours(eojeol_syl_contours[i])

            entry = {"text": ej, "idx": i, **dist.to_dict(),
                     "syllable_contours": mean_syl_contours}
            eojeol_entries.append(entry)
            print(f"  {ej!r}: {dist._n}명, {dist._mode}, {len(mean_syl_contours)}음절")

        output["sentences"][text] = {
            "n_speakers": len(wav_paths),
            "eojeols": eojeol_entries,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    print(f"\n완료 → {out_path}  ({elapsed/60:.1f}분)")
    if skipped:
        print(f"스킵 WAV {len(skipped)}개:")
        for s in skipped:
            print(f"  {s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/korean_distribution.json")
    args = parser.parse_args()
    main(Path(args.out))