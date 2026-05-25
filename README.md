uv run python3 scripts/make_intonation_from_wavs.py --learner data/learner/좋아하는음식이국수에요.wav --native data/tts/좋아하는음식이국수에요.wav \
--text "좋아하는 음식이 국수에요." --out artifacts/

  ## 출력 파일

  `scripts/make_intonation_from_wavs.py` 1회 실행 시 출력

  | 파일 | 설명 |
  |---|---|
  | `prosody.json` | Forced alignment 결과 — 학습자/원어민 음소 단위 시간 segments. 음절·어절 경계 계산의 source. |
  | `records.json` | Lens-rule outlier records — 코드가 판정한 rule label + severity + 음절 hint + 근거 metric. LLM이 NL feedback 합성 시 input. |
  | `plot_syllable_noalign.html` | 음절 단위 / 시간축 보존(비율 리샘플) / f0. 음절별 길이·타이밍 어긋남 확인. |
  | `plot_syllable_dtw.html` | 음절 단위 / DTW 모양 매칭 / f0. 음절 내 contour 일치도 (타이밍은 워프가 normalize). |
  | `plot_eojeol_noalign.html` | 어절 단위 / 시간축 보존 / f0. 어절 길이·시작·끝 비교. |
  | `plot_eojeol_noalign_delta.html` | 어절 단위 / 시간축 보존 / Δf0(피치 변화율). 음역대 차이 제거, 시간 위치별 기울기 비교. |
  | `plot_eojeol_dtw.html` | 어절 단위 / DTW / f0. 어절 contour 모양 일치도. |
  | `plot_eojeol_dtw_delta.html` | 어절 단위 / DTW / Δf0. **시간·음역대 둘 다 normalize → 순수 기울기 방향만 비교.** `pitch_rising_excess` / `pitch_falling_excess` rule의 시각화 기반. |
  | `plot_global_dtw.html` | 발화 전체 / DTW / f0. 어절 분할 없이 전체 contour 흐름. |

  축 의미 한 줄:
  - Segmenter: syllable(음절) / eojeol(어절) / global(발화 전체)
  - Aligner: noalign(원본 시간축 비율 보존) / dtw(모양만 정렬, 시간 normalize)
  - Feature: 기본 z-score f0 / delta(시간 미분, 음역대 무관 기울기)
