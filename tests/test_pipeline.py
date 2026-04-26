from __future__ import annotations

import unittest
import numpy as np
import tempfile
from pathlib import Path

from src import cost_model
from src.alignment import align_ipa_sequences, score_reference_candidates
from src.backdata_export import build_evaluation_backdata, save_evaluation_backdata, save_evaluation_bundle
from src.error_analysis import classify_alignment_errors
from src.forced_alignment import force_align_candidate
from src.ipa_utils import build_ipa_sequence
from src.korean_ipa import normalize_korean_text, pronunciation_to_ipa
from src.label_to_ipa import decode_label_text, ipa_token_to_label
from src.feedback_report import build_feedback_report
from src.quality import analyze_audio_quality, assess_alignment_confidence
from src.reference_builder import text_to_pronunciation
from src.scoring import build_score_breakdown
from src.types import AudioQualityReport, EvaluationResult, PronunciationCandidate


class PronunciationReferenceTests(unittest.TestCase):
    def test_normalize_korean_text_converts_hour_digits(self):
        self.assertEqual(normalize_korean_text("매일 아침 5시에 일어나요."), "매일 아침 다섯시에 일어나요.")

    def test_gachi_candidate_present(self):
        reference = text_to_pronunciation("같이")
        self.assertIn("가치", [candidate.pronunciation for candidate in reference.candidates])

    def test_johta_surface_candidate(self):
        reference = text_to_pronunciation("좋다")
        self.assertIn("조타", [candidate.pronunciation for candidate in reference.candidates])

    def test_meongneun_surface_candidate(self):
        reference = text_to_pronunciation("먹는")
        self.assertIn("멍는", [candidate.pronunciation for candidate in reference.candidates])

    def test_reference_ipa_is_in_true_ipa_space(self):
        sequence = pronunciation_to_ipa("안녕하세요")
        self.assertIn("ʌ", sequence.normalized_text)
        self.assertIn("ŋ", sequence.normalized_text)

    def test_reference_ipa_does_not_keep_digit_tokens(self):
        reference = text_to_pronunciation("매일 아침 5시에 일어나요.")
        self.assertNotIn("5", reference.representative_ipa.token_symbols)


class LabelToIPATests(unittest.TestCase):
    def test_model_labels_convert_to_ipa(self):
        labels, sequence = decode_label_text("N iEO NG H A S E O")
        self.assertEqual(labels[0], "N")
        self.assertIn("jʌ", sequence.normalized_text)
        self.assertIn("ŋ", sequence.normalized_text)

    def test_alignment_override_maps_we_to_supported_label(self):
        self.assertEqual(ipa_token_to_label("we"), "oE")


class CostModelTests(unittest.TestCase):
    def test_affricate_aspiration_confusion_is_small(self):
        cost, label, _, _ = cost_model.substitution_cost("tɕ", "tɕʰ", profile="ru")
        self.assertLess(cost, 0.3)
        self.assertIn("confusion", label)

    def test_vowel_confusion_lower_than_cross_category(self):
        vowel_cost, _, _, _ = cost_model.substitution_cost("ʌ", "o", profile="ru")
        cross_cost, _, _, _ = cost_model.substitution_cost("ʌ", "k", profile="ru")
        self.assertLess(vowel_cost, cross_cost)


class AlignmentTests(unittest.TestCase):
    def test_alignment_penalizes_coda_deletion_less_than_full_substitution(self):
        ref = build_ipa_sequence("m ʌ k̚").tokens
        hyp = build_ipa_sequence("m ʌ").tokens
        result = align_ipa_sequences(ref, hyp, cost_model_module=cost_model, profile="ru")
        self.assertLess(result.total_cost, 1.0)

    def test_reference_candidates_pick_surface_variant(self):
        reference = text_to_pronunciation("같이")
        hyp_tokens = reference.candidates[0].ipa.tokens
        result = score_reference_candidates(reference.candidates, hyp_tokens, cost_model_module=cost_model, profile="ru")
        self.assertEqual(result.selected_reference_candidate.pronunciation, reference.candidates[0].pronunciation)

    def test_forced_alignment_produces_segments(self):
        candidate = PronunciationCandidate("안", build_ipa_sequence("a n"), is_primary=True)
        logits = [
            [0.0, -3.0, 4.0],
            [4.0, -3.0, -2.0],
            [4.0, -2.0, -2.0],
            [-2.0, -2.0, 4.0],
            [-3.0, 4.0, -2.0],
            [-3.0, 4.0, -2.0],
        ]
        frame_timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        result = force_align_candidate(candidate, logits, frame_timestamps, {"A": 0, "N": 1, "[PAD]": 2}, blank_id=2)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].token, "a")
        self.assertGreater(result.coverage, 0.9)


class ScoringAndIssuesTests(unittest.TestCase):
    def test_score_breakdown_degrades_with_errors(self):
        candidate = PronunciationCandidate("가치", build_ipa_sequence("k a tɕʰ i"), is_primary=True)
        result = score_reference_candidates([candidate], build_ipa_sequence("k a tɕ i").tokens, cost_model_module=cost_model, profile="ru")
        score = build_score_breakdown(result)
        self.assertLess(score.overall, 100.0)
        self.assertLess(score.consonant, 100.0)

    def test_issue_classifier_detects_laryngeal_issue(self):
        candidate = PronunciationCandidate("조타", build_ipa_sequence("tɕ o tʰ a"), is_primary=True)
        result = score_reference_candidates([candidate], build_ipa_sequence("tɕ o t a").tokens, cost_model_module=cost_model, profile="ru")
        issues = classify_alignment_errors(result, profile="ru")
        self.assertTrue(any(issue.issue_type == "aspiration_or_tense_confusion" for issue in issues))


class PipelineGateTests(unittest.TestCase):
    def test_audio_quality_rejects_mostly_silent_input(self):
        audio = np.zeros(16000, dtype=np.float32)
        report = analyze_audio_quality(audio, 16000)
        self.assertFalse(report.passed)
        self.assertGreater(report.silence_ratio, 0.9)

    def test_alignment_confidence_accepts_clean_alignment(self):
        candidate = PronunciationCandidate("안", build_ipa_sequence("a n"), is_primary=True)
        logits = [
            [0.0, -3.0, 4.0],
            [4.0, -3.0, -2.0],
            [4.0, -2.0, -2.0],
            [-2.0, -2.0, 4.0],
            [-3.0, 4.0, -2.0],
            [-3.0, 4.0, -2.0],
        ]
        result = force_align_candidate(candidate, logits, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], {"A": 0, "N": 1, "[PAD]": 2}, blank_id=2)
        confidence = assess_alignment_confidence(result)
        self.assertTrue(confidence.passed)


class BackdataExportTests(unittest.TestCase):
    def test_backdata_payload_contains_handoff_sections(self):
        candidate = PronunciationCandidate("안", build_ipa_sequence("a n"), is_primary=True)
        reference = text_to_pronunciation("안")
        recognition_tokens = build_ipa_sequence("a n").tokens
        coarse = score_reference_candidates([candidate], recognition_tokens, cost_model_module=cost_model, profile="ru")
        score = build_score_breakdown(coarse)
        issues = classify_alignment_errors(coarse, profile="ru")
        feedback = build_feedback_report(issues, score, profile="ru")
        forced = force_align_candidate(
            candidate,
            [
                [4.0, -3.0, -2.0],
                [4.0, -3.0, -2.0],
                [-3.0, -2.0, 4.0],
                [-3.0, 4.0, -2.0],
                [-3.0, 4.0, -2.0],
            ],
            [0.0, 0.1, 0.2, 0.3, 0.4],
            {"A": 0, "N": 1, "[PAD]": 2},
            blank_id=2,
        )
        confidence = assess_alignment_confidence(forced)
        result = EvaluationResult(
            evaluation_status="ready",
            status_message="ok",
            reference_text=reference.normalized_text,
            reference_pronunciation="안",
            reference_ipa=build_ipa_sequence("a n"),
            reference_candidates=[candidate],
            selected_reference_candidate=coarse.selected_reference_candidate,
            user_ipa_raw="a n",
            user_ipa_normalized="a n",
            user_tokens=recognition_tokens,
            alignment_result=coarse,
            score_breakdown=score,
            feedback_report=feedback,
            quality_report=AudioQualityReport(
                passed=True,
                duration_sec=1.0,
                rms_db=-14.0,
                silence_ratio=0.1,
                clipping_ratio=0.0,
            ),
            forced_alignment_result=forced,
            alignment_confidence_report=confidence,
            debug={"raw_label_text": "A N", "raw_labels": ["A", "N"]},
        )

        payload = build_evaluation_backdata(result, profile="ru", audio_source_name="sample.wav")
        self.assertIn("llm_feedback_input", payload)
        self.assertIn("prosody_input", payload)
        self.assertIn("phoneme_segments", payload["prosody_input"])
        self.assertTrue(payload["gates"]["audio_quality_gate"]["passed"])
        self.assertTrue(payload["gates"]["coarse_token_alignment_gate"]["passed"])
        self.assertTrue(payload["gates"]["alignment_confidence_gate"]["passed"])

        with tempfile.TemporaryDirectory() as tmpdir:
            source_audio = Path(tmpdir) / "sample.wav"
            source_audio.write_bytes(b"RIFFTESTDATA")
            bundle = save_evaluation_bundle(
                result,
                profile="ru",
                audio_source_name="sample.wav",
                source_audio_path=source_audio,
                out_dir=Path(tmpdir),
            )
            path = save_evaluation_backdata(result, profile="ru", audio_source_name="sample.wav", out_dir=Path(tmpdir))
            self.assertTrue(path.exists())
            self.assertTrue(bundle["json_path"].exists())
            self.assertTrue(bundle["audio_path"].exists())
            self.assertEqual(bundle["artifact_dir"], bundle["json_path"].parent)
            self.assertEqual(bundle["artifact_dir"], bundle["audio_path"].parent)
            self.assertEqual(bundle["json_path"].stem, bundle["audio_path"].stem)


if __name__ == "__main__":
    unittest.main()
