# 억양 분석 시스템 — Design Plan

고려인(러시아어 모국어 화자)의 한국어 발화 → 한국인 원어민 억양 분포와 비교 → 위치 특정 + 오류 분류.

---

## 1. 시스템 목표

### 선택지
- (A) 점수: 학습자 발화의 단일 score
- (B) 위치 특정: 어느 음절/어절에서 억양이 틀렸는가
- (C) 오류 분류: rising/falling/flat/elongation 등 카테고리화
- (D) 교정 피드백: 자연어 학습 가이드
- (E) 연구 분석: L1 transfer 통계/분포 분석

### 선택: **(B) + (C)**

### 이유
- 교수님 피드백("러시아어권 화자의 특정 발음", "한국인 분포에서 벗어난 부분만 교정")이 (C) 카테고리 분류 방향을 강하게 시사
- 학습자에게 의미 있는 출력은 단일 점수보다 위치 + 분류 — "60점이야"보다 "둘째 음절 억양이 rising이어야 하는데 flat으로 발화"가 학습 가치가 큼
- (D) 자연어 피드백은 GPT/팀원에 위임 가능 — 우리 시스템은 (B)+(C)까지 정확히 출력
- (E)는 부산물로 자연스럽게 따라옴
- (A) 단일 점수는 위치 정보를 잃어 교수님 피드백과 충돌

---

## 2. 비교 패러다임 (Reference)

### 선택지
- (A) 현재: TTS 1개 voice = reference (1:1 pairwise)
- (B) Multi-voice TTS: 여러 TTS voice 평균 분포
- (C) 한국인 실제 화자 분포: 다양한 한국인 발화 데이터
- (D) Hybrid: TTS는 "이상적 정답" 시각화 + 한국인 분포로 "허용 범위" 정의

### 선택: **(C) → 단계적으로 (D)**

### 이유
- (A) 폐기 근거: 학습자가 비슷하게 발화했는데도 그래프가 안 비슷한 현상의 원인. 한국인도 같은 문장을 다양한 억양으로 발화 → TTS 1개로는 "정상 변동"을 정의할 수 없음. **측정 오차가 아니라 정의의 문제**
- (B) 미봉책: TTS 4개도 동일 엔진 출력이라 분포가 좁음. 자연 변동 못 잡음
- (C) 정답: 교수님 피드백 핵심 — 데이터 수집 비용은 안전장치(아래 #10)로 완화
- (D) 최종형: TTS는 시각화/직관 비교용, 판정은 분포 기준

---

## 3. 분석 단위

### 선택지
- (A) 음절 단위 (현재): 50프레임 리샘플 비교
- (B) 어절(띄어쓰기) 단위
- (C) 단어/형태소 단위
- (D) 계층적 (어절 1차 + 음절 drill-down)

### 선택: **(D) 계층적**

### 이유
- (A) 음절만으로는 부족: 한국어 억양은 음절 하나로 완결되지 않음. 앞뒤 음절과의 관계가 핵심. 그래프 noise의 진짜 원인이 fragmentation
- (C) 단어 단위는 비쌈: 형태소 분석기 의존, 단어 경계 모호, 합성어 처리 골치
- (B) 어절 단위가 sweet spot: 띄어쓰기로 이미 분리됨 + 한국어 운율의 자연스런 단위(억양구가 어절 경계에서 끊김) + 분포 통계 데이터 충분
- (D)가 최선: (B)로 어디 틀렸는지 잡고, 그 어절 안에서 (A) 음절 단위로 위치 특정. 목표 (B)+(C)에 정확히 맞음

### 구현 측면
- 어절 경계는 reference text 띄어쓰기 + forced alignment로 자동 추출
- 어절 단위 contour 비교는 50프레임 리샘플 방식 그대로 재사용
- 어절 안 음절 metric도 같이 저장 → hierarchical drill-down

---

## 4. Vector 표현

### 선택지
- (A) F0 contour 그대로 (50dim): 50프레임 z-score F0
- (B) F0 통계 요약 (~8dim): mean, std, slope, range, max/min 위치, start/end
- (C) F0 + Duration 통계 (~12dim): B + duration, syllable_count, last_syl_ratio, voiced_ratio
- (D) Multi-modal 통계 (~30dim): C + MFCC + spectral features
- (E) Wav2Vec2 embedding (~768dim): pretrained learned features

### 선택: **(C) → 데이터 보고 (D)로 확장**

### 이유
- (A) 탈락: 각 dim이 "1/50 시점"이라 의미 없음. 분포 평균은 의미 있지만 dim별 deviation이 분류로 안 이어짐 — 목표 (C) 충족 못 함
- (E) 탈락: 해석 불가. "이 dim에서 +2σ 벗어남" → 그 dim이 뭔지 모르면 사용자 피드백 못 만듦
- (B) 부족: F0만으로는 한국어 prosody 안 잡힘. duration 핵심 — 러시아어권 강세 음절 elongation 못 잡음
- (D) 이상적이지만 시작은 (C): 30dim 분포 추정에 데이터 부족 위험. 12dim으로 시작 → 분포 안 맞으면 (D) 확장

### 구체적 12dim 구성
```python
eojeol_vector = [
    f0_mean,           # 어절 평균 z-score F0
    f0_std,            # F0 변동성
    f0_slope,          # 어절 시작→끝 기울기 (rising/falling)
    f0_range,          # max - min
    f0_max_pos,        # 최고점 상대 위치 (0~1)
    f0_min_pos,        # 최저점 상대 위치 (0~1)
    f0_start,          # 시작 F0
    f0_end,            # 끝 F0
    duration,          # 어절 길이 (초)
    syllable_count,    # 음절 수
    last_syl_ratio,    # 마지막 음절 길이 / 평균 음절 길이 (러시아 강세 elongation)
    voiced_ratio,      # voiced frame / total frame
]
```

---

## 5. 시나리오 (Fixed vs Open)

### 선택지
- (Fixed) 미리 정해진 reference text 풀에서만 학습자 발화
- (Open) 임의 문장 자유 발화 가능

### 선택: **Fixed**

### 이유
- 한국어 학습용 시스템(교재/문장 풀 기반)에 자연스럽게 맞음
- 데이터 수집 비용 통제 가능 (10~50문장 × 5~30명)
- 같은 텍스트끼리 비교 → 분포 판정 정확
- Open은 한국어 운율 모델 학습 필요 → 프로젝트 범위 초과
- 향후 확장: Fixed로 신뢰도 검증 후 GPT fallback evaluator로 Open 모드 추가 가능

### 워크플로우
1. 시스템에 reference text 풀 등록
2. 각 문장마다 한국인 화자 sample 수집 → 어절별 vector 분포 구축
3. 학습자는 풀 안의 문장 중 하나 선택 → 발화 → 평가

---

## 6. 분포 단위

### 선택지
- (A) 어절 instance별: "이 reference text의 i번째 어절"의 분포
- (B) 어절 유형별: 카테고리화(품사/위치/길이) 후 카테고리별 분포
- (C) Hybrid: (A) 우선, sample 부족 시 (B) fallback

### 선택: **(A)**

### 이유
- Fixed text 가정에서 (A)가 가장 정확하고 단순
- (B)는 유형 정의 자체가 또 다른 design problem
- (C)는 데이터 부족 시점 가서 결정해도 늦지 않음

---

## 7. 분포 추정 / 판정 / 분류

### 선택지
- (A) Mahalanobis distance + threshold (Gaussian)
- (B) GMM + log-likelihood (multimodal)
- (C) KDE + density (비모수)
- (D) Per-dim z-score + rule-based
- (E) (A) + (D) 조합

### 선택: **(E)**

### 이유
- 30 sample은 12dim Gaussian 추정 한계지점 — GMM(B)/KDE(C)는 더 많은 데이터 필요
- 순수 (D) per-dim은 false positive 많음: dim 간 correlation 무시 → 자연 발화 변동을 outlier로 잘못 잡음
- (A) 단독은 분류 약함: distance만으로 "어떤 dim이 문제"가 안 나옴
- (E) 조합이 최적: Mahalanobis로 1차 판정 (correlation 고려) → 분포 밖 어절만 dim별 z-score로 분해 → rule-based 카테고리 분류

### Covariance 추정
- 정상: Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`)
- **데이터 부족 안전장치**: sample < 12면 diagonal covariance 자동 fallback

### 분류 rule mapping (6 카테고리)
```python
if z["f0_slope"] > 2:        labels.append("rising 과도 (러시아어 어말 상승 의심)")
if z["f0_slope"] < -2:       labels.append("falling 과도")
if abs(z["f0_slope"]) < 0.5 and z["f0_range"] < -2: labels.append("억양 평탄 (한국어 prosody 부족)")
if z["last_syl_ratio"] > 2:  labels.append("마지막 음절 elongation (러시아어 강세 패턴)")
if z["duration"] > 2:        labels.append("어절 전체 느림")
if z["voiced_ratio"] < -2:   labels.append("voiced 비율 낮음 (자음 과다/모음 약화)")
```
초안 — 실제 데이터 보고 fine-tune.

---

## 8. 음절 단위 Drill-down

### 선택지
- (A) 음절도 vector + 분포 적용
- (B) 음절은 현재 metric 그대로 (vs TTS 1개 비교)
- (C) 한국인 평균 contour를 음절 baseline으로

### 선택: **(C)**

### 이유
- (A) over-engineering: 음절 단위까지 분포 추정 시 sample이 더 적어 신뢰도 ↓. 짧은 음절은 vector 통계가 noisy
- (B) 현재 문제 유지: TTS 1개 reference의 한계가 음절 단위에 그대로 남음. 어절은 분포 기반으로 갔다가 음절에서 1개 voice로 돌아가면 일관성 깨짐
- (C) 자연스러움: 어절 분포 데이터로 음절 평균 contour 동시 추출 가능 (같은 녹음에서). 별도 데이터 수집 불필요

### 핵심 명확화
- **학습자 음절 경계**: 학습자 자신의 forced alignment로 추출 (한국인 경계를 학습자에 주입하지 않음)
- **음절 reference contour**: 한국인 30명의 50프레임 리샘플 F0 평균 (시간 정규화 후 frame-wise 평균)
- **음절 단위 비교**: RMSE/Pearson(학습자 contour, 한국인 평균 contour)
- **duration 정보**: 어절 vector의 다른 dim에서 잡음 — 음절은 contour 형태만

### Drill-down 출력 예시
```
어절 "안녕하세요" → Mahalanobis distance = 3.2 (분포 밖)
   분류: "마지막 음절 elongation (러시아어 강세 패턴)"
   드릴다운:
     "안" → RMSE 0.3 (정상)
     "녕" → RMSE 0.4 (정상)
     "하" → RMSE 0.5 (정상)
     "세" → RMSE 0.7 (경계)
     "요" → RMSE 2.1 (이상) ← 위치 특정
```

---

## 9. 검증 방법 + GPT 역할

### 검증 선택지
- (A) Manual gold labeling: 한국어 교사 직접 라벨링
- (B) GPT를 oracle로
- (C) Synthetic perturbation: TTS 정상 발화에 인위적 변형 주입
- (D) Coarse 자연/어색 2단계 라벨
- (E) (B) + (C) 조합

### 선택: **(E) Synthetic + GPT oracle 단계적**

### 이유
- (A) 단독 비현실적: 한국어 교사 섭외 비용 ↑
- (B) 단독 불충분: GPT가 음성 prosody에 정확한지 검증 안 됨
- (C) 선행되어야 함: Synthetic은 시스템 capability 단위 검증 ("F0 slope +2σ 변형 시 잡는가?"). 이게 안 되면 (B)/(D) 무의미
- 다음 (B): Synthetic 통과한 시스템을 실제 데이터에 적용 → GPT 분석과 일치도 측정 → 불일치는 case-by-case 판단

### GPT의 3가지 역할
1. **Evaluation oracle**: 시스템 출력 vs GPT 분석 일치도 측정 (검증용)
2. **Natural language feedback**: rule_label + per_dim_z → 학습자 친화 자연어 (**팀원 담당**)
3. **Fallback evaluator**: borderline distance(예: 1.5 < d < 2.5)에서 GPT에 양면 판정 요청

---

## 10. 데이터 수집 전략

### 선택지
- (A) 기존 데이터셋 (Common Voice / AI Hub)
- (B) 자체 녹음
- (C) Multi-voice TTS + 소규모 자체 녹음 hybrid
- (D) 데이터 수집 미루고 framework 우선

### 선택: **(D) → (A) → (B) 단계적**

### 이유
- 데이터 수집 전 framework 검증이 우선. 코드/파이프라인이 망가져 있으면 데이터 모아도 무의미. (D)로 framework 완성 + Phase 0 synthetic 검증 데이터 없이도 가능
- (A) 부분 활용: AI Hub에서 중복 많은 문장 선별 (실제 어절당 sample 수 30 미만 케이스 존재 — **데이터 부족 안전장치 필수**)
- (B) 보강은 마지막: 학과/연구실 5~10명 추가
- (C) 단독 위험: TTS 분포는 한국인 자연 변동을 대표 못 함

### 데이터 부족 안전장치
- 어절당 sample < 12: diagonal covariance만 사용 (correlation 가정 포기, 안정성 ↑)
- Per-dim z-score는 그대로 작동
- schema의 `data_quality` 메타필드로 모드 노출

---

## 11. Voiceless 부분 처리 + MFCC/Spectrum 확장

### 선택지
- (A) 12dim 유지, voiceless 무시
- (B) 12dim → 30dim 즉시 확장 (MFCC + spectral)
- (C) 단계적 확장: 12dim 검증 후 false negative에서 (D) 결정
- (D) Voiceless 단독 처리: prosody는 12dim, 자음/발음은 별도 모듈

### 선택: **(C) + (D) hybrid**

### 이유
- (B) 즉시 30dim 위험: 데이터 부족(어절당 30 미만)에서 30dim covariance 추정 거의 불가능
- (A) 단독은 교수님 피드백 무시: 자음 정보 통합 안 하면 핵심 L1 transfer 못 잡음
- (C) + (D) 조합:
  - **prosody 영역**: 12dim 어절 vector (F0/duration)
  - **자음/발음 영역**: `src/` 음소 평가 파이프라인이 담당 (음소 단위 IPA 비교 + alignment confidence)
  - **두 결과 통합**: 어절별 출력에 (prosody) + (phoneme) **별도 필드**
- MFCC/spectral 확장은 후순위: prosody 영역에 voiceless 정보가 필요한 이유는 "F0만으로 부족할 때". 12dim 검증 → false negative 분석 → 필요시 확장

---

## 12. 출력 Schema

### 선택지
- (A) Flat 구조 (모든 음절 1차원 배열)
- (B) 어절 outer + 음절 drill-down 중첩
- (C) 평행 구조 (어절/음절/어구 분리 배열)

### 선택: **(B) 중첩 구조 + per_dim_z_scores 항상 포함**

### 이유
- 시스템 자체가 어절→음절 계층 → schema가 그것을 반영해야 일관성
- 학습자 UI에서 "어절 클릭 → 음절 상세" 인터랙션 자연스러움
- 팀원 자연어 prompt에 "어절 분류 → 안의 음절 위치"가 자연스럽게 들어감
- per_dim_z 항상 포함: 디버깅 + 팀원 prompt 활용

### Schema (제안)
```json
{
  "reference_text": "오늘 티셔츠를 입고 있었어요.",
  "overall": {
    "korean_distribution_avg_distance": 1.8,
    "verdict": "natural | borderline | unnatural",
    "phoneme_accuracy_score": 0.85,
    "data_quality": {
      "covariance_mode": "diagonal | full_shrunk",
      "min_eojeol_sample_count": 8,
      "warning": "분포 sample 부족 — 결과 신뢰도 보통"
    }
  },
  "eojeol_results": [
    {
      "index": 0,
      "text": "오늘",
      "boundary": [0.0, 0.45],
      "syllable_indices": [0, 1],
      "prosody": {
        "vector": [12 floats],
        "mahalanobis_distance": 0.8,
        "in_distribution": true,
        "per_dim_z_scores": {
          "f0_mean": 0.3, "f0_std": -0.1, "f0_slope": 0.5, "f0_range": 0.2,
          "f0_max_pos": 0.0, "f0_min_pos": -0.4, "f0_start": 0.1, "f0_end": -0.2,
          "duration": 0.6, "syllable_count": 0.0, "last_syl_ratio": 0.4, "voiced_ratio": -0.1
        },
        "rule_labels": []
      },
      "phoneme": {
        "alignment_confidence": 0.92,
        "issues": []
      },
      "syllable_drilldown": [
        {
          "index": 0, "label": "오", "boundary": [0.0, 0.22],
          "contour": {
            "learner_f0": [50 floats],
            "korean_mean_f0": [50 floats],
            "joint_voiced_mask": [50 bools]
          },
          "metrics": {"rmse": 0.3, "pearson": 0.85, "slope_diff": 0.1, "voiced_frame_count": 18},
          "is_outlier": false
        }
      ]
    }
  ],
  "tts_reference": {
    "wav": "artifacts/tts_cache/xxx.wav",
    "phoneme_segments": [],
    "syllable_boundaries": []
  }
}
```

---

## 13. 모듈 구조

### 선택지
- (A) 현재 `core/` 유지 + 새 모듈 추가
- (B) `core/intonation/` 서브패키지 그룹화
- (C) `core/`를 `prosody/`로 rename

### 선택: **(A)**

### 이유
- 프로젝트 진행 중반 — 큰 구조 변경 비용 ↑
- 새 기능은 새 파일로 명확히 분리 가능
- import 경로 안정

### 구조
```
core/
├── f0_extractor.py       (기존)
├── comparator.py         (확장: 한국인 평균 contour 비교)
├── syllable_utils.py     (기존)
├── metrics.py            (기존)
├── plotter.py            (확장: schema 기반)
├── tts.py                (기존, 시각화 전용)
├── eojeol_vector.py      (신규: 12dim vector 추출)
├── distribution.py       (신규: Mahalanobis + z-score + rule mapping)
├── integrator.py         (신규: prosody + phoneme(src/) 통합)
└── synthetic.py          (신규: Phase 0 perturbation)
```

---

## 14. Phase 마일스톤

| Phase | 데이터 | 산출물 | 통과 기준 |
|---|---|---|---|
| 0 | 0개 | framework + synthetic 검증 | precision/recall ≥ 0.7 |
| 1 | 5~10문장 × 5~10명 (AI Hub) | 분포 fitting + 음절 평균 contour | false positive < 30% |
| 2 | 러시아어권 30~50개 | rule label confusion matrix | GPT 일치도 kappa ≥ 0.5 |
| 3 | 30~50문장 × 20~30명 | full covariance, MFCC 확장 검토 | 팀원 NL feedback 통합 |

---

## 15. 폐기/유보 결정

- **DTW 폐기**: 분포 기반 + vector화 + 음절 평균 contour 비교는 alignment 자유로움. DTW로 회귀하지 않음
- **MFCC/spectral**: Phase 3 이후 false negative 분석 결과 보고 결정
- **자음/voiceless 평가**: `src/` 음소 평가 모듈에 위임, 결과만 schema의 `phoneme` 필드로 통합
- **TTS**: 판정 로직에서 완전히 분리, 시각화 전용