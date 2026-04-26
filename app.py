from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src import cost_model
from src.alignment import score_reference_candidates
from src.audio_to_ipa import AudioToIPARecognizer
from src.backdata_export import save_evaluation_bundle
from src.error_analysis import classify_alignment_errors
from src.feedback_report import build_feedback_report
from src.forced_alignment import force_align_candidate
from src.quality import assess_alignment_confidence
from src.recognition import recognize_audio
from src.reference_builder import text_to_pronunciation
from src.scoring import build_score_breakdown
from src.types import EvaluationResult
from src.ui_helpers import alignment_rows, short_evaluation_line


st.set_page_config(page_title="한국어 IPA 발음 평가", layout="wide")
COARSE_SIMILARITY_THRESHOLD = 25.0


@st.cache_resource
def load_recognizer() -> AudioToIPARecognizer:
    return AudioToIPARecognizer()


def save_uploaded_audio_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix if uploaded_file.name else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def finalize_evaluation_result(result: EvaluationResult, *, profile: str, audio_file, source_audio_path: str) -> EvaluationResult:
    audio_source_name = getattr(audio_file, "name", None) or "microphone_input.wav"
    artifact_paths = save_evaluation_bundle(
        result,
        profile=profile,
        audio_source_name=audio_source_name,
        source_audio_path=source_audio_path,
    )
    result.debug["artifact_dir"] = str(artifact_paths["artifact_dir"].resolve())
    result.debug["json_path"] = str(artifact_paths["json_path"].resolve())
    if artifact_paths["audio_path"] is not None:
        result.debug["audio_artifact_path"] = str(artifact_paths["audio_path"].resolve())
    return result


def evaluate_audio(target_text: str, audio_file, profile: str = "ru") -> EvaluationResult:
    recognizer = load_recognizer()
    temp_audio_path = save_uploaded_audio_to_temp(audio_file)
    try:
        def persist(result: EvaluationResult) -> EvaluationResult:
            return finalize_evaluation_result(
                result,
                profile=profile,
                audio_file=audio_file,
                source_audio_path=temp_audio_path,
            )

        reference = text_to_pronunciation(target_text)
        default_candidate = reference.candidates[0]
        recognition = recognize_audio(recognizer, temp_audio_path)

        if recognition.quality_report is not None and not recognition.quality_report.passed:
            return persist(EvaluationResult(
                evaluation_status="retry",
                status_message="음성 품질이 낮아 평가를 진행하지 않았습니다. 재녹음을 권장합니다.",
                reference_text=reference.normalized_text,
                reference_pronunciation=reference.representative_pronunciation,
                reference_ipa=reference.representative_ipa,
                reference_candidates=reference.candidates,
                selected_reference_candidate=default_candidate,
                user_ipa_raw=recognition.raw_text,
                user_ipa_normalized=recognition.normalized_text,
                user_tokens=recognition.tokens,
                alignment_result=None,
                score_breakdown=None,
                feedback_report=None,
                quality_report=recognition.quality_report,
                debug={
                    "audio_quality_gate_passed": False,
                    "coarse_token_alignment_gate_passed": None,
                    "alignment_confidence_gate_passed": None,
                    "raw_label_text": recognition.raw_label_text,
                    "raw_labels": recognition.raw_labels,
                },
            ))

        if not recognition.tokens:
            return persist(EvaluationResult(
                evaluation_status="retry",
                status_message="사용자 음성에서 IPA/phone 시퀀스를 안정적으로 추정하지 못했습니다.",
                reference_text=reference.normalized_text,
                reference_pronunciation=reference.representative_pronunciation,
                reference_ipa=reference.representative_ipa,
                reference_candidates=reference.candidates,
                selected_reference_candidate=default_candidate,
                user_ipa_raw=recognition.raw_text,
                user_ipa_normalized=recognition.normalized_text,
                user_tokens=recognition.tokens,
                alignment_result=None,
                score_breakdown=None,
                feedback_report=None,
                quality_report=recognition.quality_report,
                debug={
                    "audio_quality_gate_passed": recognition.quality_report.passed if recognition.quality_report is not None else None,
                    "coarse_token_alignment_gate_passed": None,
                    "alignment_confidence_gate_passed": None,
                },
            ))

        coarse_alignment = score_reference_candidates(reference.candidates, recognition.tokens, cost_model_module=cost_model, profile=profile)
        if coarse_alignment.normalized_score < COARSE_SIMILARITY_THRESHOLD:
            return persist(EvaluationResult(
                evaluation_status="retry",
                status_message="정답 문장과 실제 발화의 유사도가 너무 낮아 forced alignment를 생략했습니다. 같은 문장을 다시 또렷하게 읽어 주세요.",
                reference_text=reference.normalized_text,
                reference_pronunciation=reference.representative_pronunciation,
                reference_ipa=reference.representative_ipa,
                reference_candidates=reference.candidates,
                selected_reference_candidate=coarse_alignment.selected_reference_candidate,
                user_ipa_raw=recognition.raw_text,
                user_ipa_normalized=recognition.normalized_text,
                user_tokens=recognition.tokens,
                alignment_result=coarse_alignment,
                score_breakdown=None,
                feedback_report=None,
                quality_report=recognition.quality_report,
                debug={
                    "audio_quality_gate_passed": recognition.quality_report.passed if recognition.quality_report is not None else None,
                    "coarse_token_alignment_gate_passed": False,
                    "alignment_confidence_gate_passed": None,
                    "coarse_similarity": coarse_alignment.normalized_score,
                    "coarse_similarity_threshold": COARSE_SIMILARITY_THRESHOLD,
                    "raw_label_text": recognition.raw_label_text,
                    "raw_labels": recognition.raw_labels,
                },
            ))

        tokenizer = getattr(recognizer.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Forced alignment에 필요한 tokenizer 정보를 찾지 못했습니다.")
        label_to_id = tokenizer.get_vocab()
        blank_id = tokenizer.pad_token_id
        try:
            forced_alignment = force_align_candidate(
                coarse_alignment.selected_reference_candidate,
                recognition.logits or [],
                recognition.frame_timestamps or [],
                label_to_id,
                blank_id,
            )
        except ValueError as exc:
            return persist(EvaluationResult(
                evaluation_status="discarded",
                status_message="기준 발음에 alignment용으로 지원되지 않는 기호가 포함되어 평가를 중단했습니다.",
                reference_text=reference.normalized_text,
                reference_pronunciation=reference.representative_pronunciation,
                reference_ipa=reference.representative_ipa,
                reference_candidates=reference.candidates,
                selected_reference_candidate=coarse_alignment.selected_reference_candidate,
                user_ipa_raw=recognition.raw_text,
                user_ipa_normalized=recognition.normalized_text,
                user_tokens=recognition.tokens,
                alignment_result=coarse_alignment,
                score_breakdown=None,
                feedback_report=None,
                quality_report=recognition.quality_report,
                debug={
                    "audio_quality_gate_passed": recognition.quality_report.passed if recognition.quality_report is not None else None,
                    "coarse_token_alignment_gate_passed": True,
                    "alignment_confidence_gate_passed": None,
                    "coarse_similarity": coarse_alignment.normalized_score,
                    "coarse_similarity_threshold": COARSE_SIMILARITY_THRESHOLD,
                    "raw_label_text": recognition.raw_label_text,
                    "raw_labels": recognition.raw_labels,
                    "alignment_error": str(exc),
                },
            ))
        confidence_report = assess_alignment_confidence(forced_alignment)
        if not confidence_report.passed:
            return persist(EvaluationResult(
                evaluation_status="discarded",
                status_message="forced alignment 신뢰도가 낮아 평가 결과를 폐기했습니다. 재녹음을 권장합니다.",
                reference_text=reference.normalized_text,
                reference_pronunciation=reference.representative_pronunciation,
                reference_ipa=reference.representative_ipa,
                reference_candidates=reference.candidates,
                selected_reference_candidate=coarse_alignment.selected_reference_candidate,
                user_ipa_raw=recognition.raw_text,
                user_ipa_normalized=recognition.normalized_text,
                user_tokens=recognition.tokens,
                alignment_result=coarse_alignment,
                score_breakdown=None,
                feedback_report=None,
                quality_report=recognition.quality_report,
                forced_alignment_result=forced_alignment,
                alignment_confidence_report=confidence_report,
                debug={
                    "audio_quality_gate_passed": recognition.quality_report.passed if recognition.quality_report is not None else None,
                    "coarse_token_alignment_gate_passed": True,
                    "alignment_confidence_gate_passed": False,
                    "coarse_similarity": coarse_alignment.normalized_score,
                    "coarse_similarity_threshold": COARSE_SIMILARITY_THRESHOLD,
                    "raw_label_text": recognition.raw_label_text,
                    "raw_labels": recognition.raw_labels,
                },
            ))

        score_breakdown = build_score_breakdown(coarse_alignment)
        issues = classify_alignment_errors(coarse_alignment, profile=profile)
        feedback_report = build_feedback_report(issues, score_breakdown, profile=profile)

        return persist(EvaluationResult(
            evaluation_status="ready",
            status_message="음질, 문장 유사도, forced alignment 신뢰도를 모두 통과했습니다.",
            reference_text=reference.normalized_text,
            reference_pronunciation=reference.representative_pronunciation,
            reference_ipa=reference.representative_ipa,
            reference_candidates=reference.candidates,
            selected_reference_candidate=coarse_alignment.selected_reference_candidate,
            user_ipa_raw=recognition.raw_text,
            user_ipa_normalized=recognition.normalized_text,
            user_tokens=recognition.tokens,
            alignment_result=coarse_alignment,
            score_breakdown=score_breakdown,
            feedback_report=feedback_report,
            quality_report=recognition.quality_report,
            forced_alignment_result=forced_alignment,
            alignment_confidence_report=confidence_report,
            debug={
                "audio_quality_gate_passed": recognition.quality_report.passed if recognition.quality_report is not None else None,
                "coarse_token_alignment_gate_passed": True,
                "alignment_confidence_gate_passed": True,
                "coarse_similarity": coarse_alignment.normalized_score,
                "coarse_similarity_threshold": COARSE_SIMILARITY_THRESHOLD,
                "raw_label_text": recognition.raw_label_text,
                "raw_labels": recognition.raw_labels,
                "frame_confidence_preview": recognition.frame_confidence[:10] if recognition.frame_confidence else [],
                "frame_timestamp_preview": recognition.frame_timestamps[:10] if recognition.frame_timestamps else [],
                "selected_candidate_notes": coarse_alignment.selected_reference_candidate.notes,
            },
        ))
    finally:
        try:
            Path(temp_audio_path).unlink(missing_ok=True)
        except Exception:
            pass


st.title("한국어 IPA 발음 평가")
st.write("정답 문장을 기준으로 발음형 후보와 IPA 기준열을 만들고, 사용자 음성을 정렬 기반으로 평가합니다.")

target_text = st.text_area("정답 문장", placeholder="예: 저는 오늘 학교에 갑니다", height=100)
profile = st.selectbox(
    "학습자 언어권 프로필",
    options=["ru", "default"],
    index=0,
    help="현재는 러시아어권 학습자를 기본값으로 두고, 일부 혼동쌍 비용을 조정합니다.",
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("마이크 녹음")
    mic_audio = st.audio_input("마이크로 녹음", sample_rate=16000)
with col2:
    st.subheader("오디오 파일 업로드")
    uploaded_audio = st.file_uploader("wav/mp3/m4a/ogg 업로드", type=["wav", "mp3", "m4a", "ogg", "flac"])

audio_source = mic_audio if mic_audio is not None else uploaded_audio
audio_source_label = "마이크" if mic_audio is not None else "파일 업로드" if uploaded_audio is not None else "없음"

if audio_source is not None:
    st.audio(audio_source)

run = st.button("발음 평가 실행", type="primary")

if run:
    if not target_text.strip():
        st.error("정답 문장을 입력해 주세요.")
    elif audio_source is None:
        st.error("마이크 녹음 또는 파일 업로드 중 하나가 필요합니다.")
    else:
        try:
            with st.spinner("기준 발음 후보 생성, 음성 분석, 정렬 평가를 수행하는 중..."):
                result = evaluate_audio(target_text, audio_source, profile=profile)

            st.subheader("A. 입력 요약")
            st.write(f"기준 문장: {result.reference_text}")
            st.write(f"선택된 입력 소스: {audio_source_label}")

            st.subheader("B. 기준 발음")
            st.write(f"표준 문장: {result.reference_text}")
            st.write(f"대표 발음형: {result.reference_pronunciation}")
            st.write(f"대표 기준 IPA: {result.reference_ipa.normalized_text}")
            st.write(f"허용 발음 후보 수: {len(result.reference_candidates)}")
            st.write(f"선택된 정렬 후보: {result.selected_reference_candidate.pronunciation}")

            st.subheader("C. 사용자 분석")
            st.write(f"사용자 추정 IPA(raw): {result.user_ipa_raw}")
            st.write(f"사용자 추정 IPA(normalized): {result.user_ipa_normalized}")
            st.code(" ".join(token.symbol for token in result.user_tokens))

            st.subheader("D. 평가 상태")
            if result.quality_report is not None:
                st.write(
                    f"음질 검사: {'통과' if result.quality_report.passed else '실패'} | "
                    f"길이 {result.quality_report.duration_sec:.2f}s | "
                    f"무음 비율 {result.quality_report.silence_ratio:.2f} | "
                    f"clipping {result.quality_report.clipping_ratio:.3f}"
                )
            if result.alignment_result is not None:
                st.write(f"거친 유사도 점수: {result.alignment_result.normalized_score:.1f}")
            if result.alignment_confidence_report is not None:
                st.write(
                    f"forced alignment 신뢰도: {'통과' if result.alignment_confidence_report.passed else '실패'} | "
                    f"coverage {result.alignment_confidence_report.coverage:.2f} | "
                    f"avg confidence {result.alignment_confidence_report.avg_token_confidence:.2f}"
                )
            st.info(result.status_message)
            if result.debug.get("artifact_dir"):
                st.write(f"산출물 폴더: {result.debug['artifact_dir']}")
            if result.debug.get("json_path"):
                st.write(f"분석 JSON 저장 경로: {result.debug['json_path']}")
            if result.debug.get("audio_artifact_path"):
                st.write(f"저장된 음성 파일 경로: {result.debug['audio_artifact_path']}")

            if result.evaluation_status == "ready":
                st.subheader("E. 점수 요약")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("전체 점수", f"{result.score_breakdown.overall:.1f}")
                m2.metric("자음 점수", f"{result.score_breakdown.consonant:.1f}")
                m3.metric("모음 점수", f"{result.score_breakdown.vowel:.1f}")
                m4.metric("받침 점수", f"{result.score_breakdown.coda:.1f}")
                st.info(short_evaluation_line(result.feedback_report))

                st.subheader("F. 주요 피드백")
                st.write(result.feedback_report.headline)
                for idx, issue in enumerate(result.feedback_report.issues, start=1):
                    st.markdown(f"**{idx}. {issue.description}**")
                    st.write(issue.tip)
                if result.feedback_report.tips:
                    st.write("개선 팁:")
                    for tip in result.feedback_report.tips[:5]:
                        st.write(f"- {tip}")

                st.subheader("G. 정렬/비교 상세")
                st.dataframe(alignment_rows(result.alignment_result), use_container_width=True)
                if result.forced_alignment_result is not None:
                    st.write("forced alignment 음소 구간:")
                    st.dataframe(
                        [
                            {
                                "음소": segment.token,
                                "label": segment.label,
                                "시작(초)": round(segment.start_time, 3),
                                "끝(초)": round(segment.end_time, 3),
                                "신뢰도": round(segment.confidence, 3),
                            }
                            for segment in result.forced_alignment_result.segments
                        ],
                        use_container_width=True,
                    )

                st.subheader("H. 점수 설명")
                st.write("전체 점수는 기준 IPA와 사용자 IPA의 weighted alignment 비용을 바탕으로 계산됩니다.")
                st.write("비슷한 소리끼리의 치환은 작은 감점, 다른 계열 소리 치환은 큰 감점으로 처리됩니다.")
                st.write("받침 누락과 경음/기식 구분 오류는 별도 비용 규칙으로 반영됩니다.")
            else:
                st.warning("평가를 진행하지 않았습니다. 입력 품질 또는 alignment 신뢰도를 먼저 확인해 주세요.")

            with st.expander("디버그 정보"):
                st.json(
                    {
                        "raw_cost": result.score_breakdown.raw_cost if result.score_breakdown is not None else None,
                        "max_cost": result.score_breakdown.max_cost if result.score_breakdown is not None else None,
                        "penalty_summary": result.score_breakdown.penalty_summary if result.score_breakdown is not None else [],
                        "debug": result.debug,
                        "alignment_feature_penalties": result.alignment_result.feature_penalties if result.alignment_result is not None else {},
                        "candidate_notes": [candidate.notes for candidate in result.reference_candidates],
                        "quality_report": result.quality_report.__dict__ if result.quality_report is not None else None,
                        "alignment_confidence_report": result.alignment_confidence_report.__dict__ if result.alignment_confidence_report is not None else None,
                    }
                )
        except Exception as exc:
            st.exception(exc)
