  ## 실행

  ### 학습자 + 원어민 wav 둘 다 있을 때
  ```
  uv run python3 scripts/make_intonation_from_wavs.py --learner data/learner/좋아하는음식이국수에요.wav --native data/tts/좋아하는음식이국수에요.wav \
  --text "좋아하는 음식이 국수에요." --out artifacts/
  ```

  ### 원어민 녹음이 없을 때 (gTTS로 자동 합성)
  ```
  uv run python3 scripts/make_intonation_with_tts.py --learner data/learner/좋아하는음식이국수에요.wav \
  --text "좋아하는 음식이 국수에요." --out artifacts/ [--tts-dir data/tts]
  ```
  내부적으로 gTTS로 native wav를 합성(`--tts-dir`에 캐시)한 뒤 `make_intonation_from_wavs.run()`을 호출하므로 출력 파일은 동일.

  ## 출력 파일

  `scripts/make_intonation_from_wavs.py` 1회 실행 시 출력 (with_tts도 동일)

  | 파일 | 설명 |
  |---|---|
  | `prosody.json` | Forced alignment 결과 — 학습자/원어민 음소 단위 시간 segments. 음절·어절 경계 계산의 source. |
  | `plot_syllable_noalign.html` | 음절 단위 / 시간축 보존(비율 리샘플) / f0. 음절별 길이·타이밍 어긋남 확인. |
  | `plot_syllable_dtw.html` | 음절 단위 / DTW 모양 매칭 / f0. 음절 내 contour 일치도 (타이밍은 워프가 normalize). |
  | `plot_eojeol_noalign.html` | 어절 단위 / 시간축 보존 / f0. 어절 길이·시작·끝 비교. |
  | `plot_eojeol_noalign_delta.html` | 어절 단위 / 시간축 보존 / Δf0(피치 변화율). 음역대 차이 제거, 시간 위치별 기울기 비교. |
  | `plot_eojeol_dtw.html` | 어절 단위 / DTW / f0. 어절 contour 모양 일치도. |
  | **`plot_eojeol_dtw_delta.html`** | 어절 단위 / DTW / Δf0. **시간·음역대 둘 다 normalize → 순수 기울기 방향만 비교.** |
  | `plot_global_dtw.html` | 발화 전체 / DTW / f0. 어절 분할 없이 전체 contour 흐름. |
  | `plot_f0f1f2_global.html` | 발화 전체 / **F1+F2 multivariate DTW** path 위에 F0 lookup. 모음 정체성 시간 기준으로 본 피치 — 같은 모음을 발음하는 순간의 F0가 어땠는가. |
  | **`plot_f0f1f2_eojeol.html`** | 위와 동일, **어절별 row 분리**. 어절 zoom-in (모음 정체성 기준). |
  | `plot_mfcc_global.html` | 발화 전체 / **MFCC c1~c12 DTW** path + F0 overlay (no-norm). articulation 채널, 화자 차이 미보정 baseline. |
  | **`plot_mfcc_cmvn_global.html`** | 위 + **per-utterance CMVN** (계수별 평균/표준편차 정규화). 화자/채널 차이 제거 후 alignment 비교 — no-norm과 나란히 보면 normalization 효과 보임. |

  축 의미 한 줄:
  - Segmenter: syllable(음절) / eojeol(어절) / global(발화 전체)
  - Aligner: noalign(원본 시간축 비율 보존) / dtw(모양만 정렬, 시간 normalize)
  - Feature: 기본 z-score f0 / delta(시간 미분, 음역대 무관 기울기)
  - Channel: f0(피치) / formant(F1+F2 정렬 기준 = 모음 정체성 시간) / mfcc(c1~c12 정렬 기준 = 전체 spectral envelope, articulation)
  - Normalization (mfcc만): no-norm / cmvn(per-utterance 평균/표준편차)
