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

# NOTE: core/* (parselmouth·dtaidistance) 임포트는 run() 안에서 torch 모델
# 구성 이후로 지연한다. parselmouth가 torch보다 먼저 로드되면 Wav2Vec2의
# weight_norm(LAPACK 호출)이 macOS arm64에서 segfault한다.
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


def _write_combined_html(
    figures: list[tuple[str, "go.Figure"]],
    title: str,
    out_path: Path,
) -> None:
    sections = []
    for i, (name, fig) in enumerate(figures):
        div = fig.to_html(full_html=False, include_plotlyjs=(i == 0))
        heading = (
            f'<h2 style="margin-top:2em;font-family:sans-serif;'
            f'border-bottom:1px solid #ccc;padding-bottom:.3em">{name}</h2>'
        )
        sections.append(heading + "\n" + div)

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="ko">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{title} — 통합 분석</title>\n"
        "<style>"
        "body{margin:24px;font-family:sans-serif}"
        "h1{border-bottom:2px solid #333;padding-bottom:.4em}"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        f"<h1>{title} — prosody 분석 통합 뷰</h1>\n"
        + "\n".join(sections)
        + "\n</body>\n</html>"
    )
    out_path.write_text(html, encoding="utf-8")


def run(
    learner_wav: Path,
    native_wav: Path,
    text: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 구두점은 pronunciation_to_ipa의 position 할당을 깨뜨리므로 사전 제거
    # _forced_align과 positions 계산이 동일한 텍스트를 사용해야 segments 수가 일치함
    clean_text = "".join(c for c in text if c not in ".·,!?。")

    print("모델 로딩 중...")
    recognizer = AudioToIPARecognizer()

    # torch 모델 구성 완료 후에만 parselmouth/dtaidistance 로드 (segfault 회피)
    from core import formants, mfcc
    from core.aligner import DtwAligner, NoAligner
    from core.f0_extractor import extract_f0
    from core.feedback import build_payload as build_records_payload
    from core.features import delta_f0
    from core.lens import build_plot_model
    from core.plotter import figure_from_model
    from core.rules import evaluate as evaluate_rules
    from core.segmenter import EojeolSegmenter, SyllableSegmenter, WholeSegmenter

    print("learner forced alignment 중...")
    learner_fa = _forced_align(recognizer, learner_wav, clean_text)

    print("native forced alignment 중...")
    native_fa = _forced_align(recognizer, native_wav, clean_text)

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

    # ── F0 추출 ──────────────────────────────────────────────────────────────
    native_f0 = extract_f0(native_wav)
    learner_f0 = extract_f0(learner_wav)
    positions = [t.syllable_position for t in pronunciation_to_ipa(clean_text).tokens]
    n_seg = payload["native"]["phoneme_segments"]
    l_seg = payload["learner"]["phoneme_segments"]
    syllable_labels = [c for c in text if c.strip() and c not in ".·,!?。"]

    # ── 렌즈 매트릭스 = Segmenter × Aligner × feature ────────────────────────
    # 새 렌즈가 보고 싶으면 dict 한 줄 추가. feature 생략 시 z-score f0.
    delta_kw = dict(
        feature=delta_f0,
        y_axis_title="ΔF0 (z-score/frame)",
        y_range=None,  # delta는 진폭이 달라 autoscale
    )
    lenses = [
        dict(name="syllable_noalign",
             segmenter=SyllableSegmenter(n_seg, l_seg, positions, syllable_labels),
             aligner=NoAligner()),
        dict(name="syllable_dtw",
             segmenter=SyllableSegmenter(n_seg, l_seg, positions, syllable_labels),
             aligner=DtwAligner()),
        dict(name="eojeol_noalign",
             segmenter=EojeolSegmenter(n_seg, l_seg, positions, clean_text),
             aligner=NoAligner()),
        dict(name="eojeol_noalign_delta",
             segmenter=EojeolSegmenter(n_seg, l_seg, positions, clean_text),
             aligner=NoAligner(), **delta_kw),
        dict(name="eojeol_dtw",
             segmenter=EojeolSegmenter(n_seg, l_seg, positions, clean_text),
             aligner=DtwAligner()),
        dict(name="eojeol_dtw_delta",
             segmenter=EojeolSegmenter(n_seg, l_seg, positions, clean_text),
             aligner=DtwAligner(), **delta_kw),
        dict(name="global_dtw",
             segmenter=WholeSegmenter(native_f0, learner_f0),
             aligner=DtwAligner()),
    ]
    generated_figs: list[tuple[str, "go.Figure"]] = []

    for cfg in lenses:
        name = cfg.pop("name")
        model = build_plot_model(
            native_f0, learner_f0, title=f"{text} — {name}", **cfg
        )
        if model is None:
            print(f"렌즈 {name}: 경계 없음, 건너뜀")
            continue
        plot_path = out_dir / f"plot_{name}.html"
        fig = figure_from_model(model)
        fig.write_html(plot_path)
        generated_figs.append((name, fig))
        print(f"plot 저장 → {plot_path}")

    # ── Records (lens-rule paradigm) ─────────────────────────────────────────
    # Segmenter 인스턴스를 rules용으로 한 번 더 만든다 — 매트릭스 내부 인스턴스와 분리.
    eojeol_seg = EojeolSegmenter(n_seg, l_seg, positions, clean_text)
    syllable_seg = SyllableSegmenter(n_seg, l_seg, positions, syllable_labels)

    # ── F0/F1/F2 시각화 (record/rule 무관 — 모음 정체성 + 피치 한 화면 비교) ─
    # F1+F2 DTW로 alignment, F0는 그 시간 좌표 위에 함께 그림 (redundant 의도 —
    # 기존 plot_eojeol_dtw_delta 등 F0 자체 alignment lens는 그대로 유지).
    learner_formants = formants.extract_formants(learner_wav)
    native_formants = formants.extract_formants(native_wav)
    f0f1f2_global_path = out_dir / "plot_f0f1f2_global.html"
    fig_f0f1f2_global = formants.build_global_figure(
        learner_formants, native_formants, learner_f0, native_f0,
        title=f"{text} — F0/F1/F2 global (F1+F2 DTW 정렬)",
    )
    fig_f0f1f2_global.write_html(f0f1f2_global_path)
    generated_figs.append(("f0f1f2_global", fig_f0f1f2_global))
    print(f"plot 저장 → {f0f1f2_global_path}")

    f0f1f2_eojeol_path = out_dir / "plot_f0f1f2_eojeol.html"
    fig_f0f1f2_eojeol = formants.build_eojeol_figure(
        learner_formants, native_formants, learner_f0, native_f0,
        eojeol_native_spans=eojeol_seg.native_spans(),
        eojeol_learner_spans=eojeol_seg.learner_spans(),
        eojeol_labels=eojeol_seg.labels(),
        title=f"{text} — F0/F1/F2 어절별 (F1+F2 DTW 정렬)",
    )
    fig_f0f1f2_eojeol.write_html(f0f1f2_eojeol_path)
    generated_figs.append(("f0f1f2_eojeol", fig_f0f1f2_eojeol))
    print(f"plot 저장 → {f0f1f2_eojeol_path}")

    # ── MFCC + F0 overlay (no-norm vs CMVN — speaker normalization 효과 비교) ─
    # MFCC c1~c12 multivariate DTW path 위 F0 lookup. articulation 채널.
    learner_mfcc = mfcc.extract_mfcc(learner_wav)
    native_mfcc = mfcc.extract_mfcc(native_wav)
    for normalize, suffix in ((False, "global"), (True, "cmvn_global")):
        mfcc_path = out_dir / f"plot_mfcc_{suffix}.html"
        fig_mfcc = mfcc.build_global_figure(
            learner_mfcc, native_mfcc, learner_f0, native_f0,
            normalize=normalize,
            title=f"{text} — MFCC + F0 ({'CMVN' if normalize else 'no-norm'})",
        )
        fig_mfcc.write_html(mfcc_path)
        generated_figs.append((f"mfcc_{suffix}", fig_mfcc))
        print(f"plot 저장 → {mfcc_path}")

    combined_path = out_dir / "plot_combined.html"
    _write_combined_html(list(reversed(generated_figs)), text, combined_path)
    print(f"통합 plot 저장 → {combined_path}")

    records = evaluate_rules(
        native_f0, learner_f0,
        eojeol_native_spans=eojeol_seg.native_spans(),
        eojeol_learner_spans=eojeol_seg.learner_spans(),
        eojeol_labels=eojeol_seg.labels(),
        syllable_native_spans=syllable_seg.native_spans(),
        syllable_learner_spans=syllable_seg.learner_spans(),
        syllable_labels=syllable_seg.labels(),
        eojeol_text=clean_text,
    )
    records_payload = build_records_payload(records, text)
    records_path = out_dir / "records.json"
    records_path.write_text(
        json.dumps(records_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"records 저장 → {records_path} ({len(records)}개)")


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