# Prosody Analysis API — 설계자 참고 문서

## 개요

학습자(고려인·러시아어 L1)의 한국어 발화를 원어민 TTS와 비교하여
음절별 억양 불일치를 정량화한다.

음소 평가 파이프라인(`pronunciation_backend_pipeline`)이 먼저 실행되고,
그 결과로 나온 `prosody_input` dict를 이 모듈이 받아 억양 분석을 수행한다.

```
[음소분석 파이프라인]                        [억양 분석 모듈]
evaluate_pronunciation_file()
  └─ build_backend_payload()
       └─ prosody_input dict  ──────────▶  analyze(prosody_input)
                                               └─ list[dict]  ──▶ API 응답
```

---

## 신경 쓸 범위

**`analyze.py`의 `analyze()` 함수만 호출하면 된다.** 나머지는 블랙박스로 취급해도 무방하다.

| 모듈 | 역할 | 개입 여부 |
|---|---|---|
| `analyze.py` | 억양 분석 진입점. **API 설계자가 호출하는 유일한 인터페이스** | **직접 호출** |
| `core/` | F0 추출, 음절 정렬, 메트릭 계산, 시각화. `analyze()`가 내부적으로 사용 | 불필요 |
| `scripts/` | 개발·디버깅용 보조 스크립트 (artifact JSON 생성 등) | 불필요 |
| `tests/test_core.py` | `core/` 단위 테스트, plot 시각화 | 불필요 |
| `conftest.py` | pytest 경로 설정 | 불필요 |

---

## 호출 조건 — 게이트 통과 여부 확인 필수

음소분석 파이프라인은 평가 전에 두 단계 게이트를 적용한다.

| 게이트 | 설명 | 탈락 시 status |
|---|---|---|
| 음질 게이트 | 너무 짧거나 무음에 가까운 오디오 조기 반환 | `"quality_fail"` |
| 정렬 게이트 | 정답과 전혀 다른 문장을 읽은 경우 탈락 | `"alignment_fail"` |

**`evaluation_status == "ready"` 일 때만 `analyze()`를 호출해야 한다.**
게이트 탈락 상태에서 호출하면 `phoneme_segments`가 비어 있어 억양 분석이 무의미하다.

---

## 호출 방법

```python
from pronunciation_backend_pipeline import get_prosody_input
from analyze import analyze

prosody_input = get_prosody_input(audio_path, reference_text)
results = analyze(prosody_input)
```

또는 `evaluate_pronunciation_file()`의 전체 응답에서 꺼낼 수도 있다.

```python
from pronunciation_backend_pipeline import evaluate_pronunciation_file
from analyze import analyze

response = evaluate_pronunciation_file(audio_path, reference_text)

if response["status"]["evaluation_status"] == "ready":
    results = analyze(response["prosody_input"])
```

---

## 함수 시그니처

```python
def analyze(
    prosody_input: dict,
    *,
    recognizer: AudioToIPARecognizer | None = None,
    tts_cache_dir: str | Path | None = None,
) -> list[dict]:
```

### 파라미터

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `prosody_input` | `dict` | `get_prosody_input()` 또는 `build_backend_payload()["prosody_input"]`의 반환값. 반드시 `audio_file_path` 키를 포함해야 한다. |
| `recognizer` | `AudioToIPARecognizer \| None` | Wav2Vec2 모델 인스턴스. **일반적으로 `None`으로 두면 된다** — 내부에서 `get_default_recognizer()` 싱글톤을 사용하므로 팀원 파이프라인과 모델을 공유한다. 테스트 목적으로 직접 주입할 때만 사용. |
| `tts_cache_dir` | `str \| Path \| None` | TTS WAV 캐시 디렉토리 경로. None이면 기본값 `artifacts/tts_cache`를 사용한다. 프로덕션에서는 절대 경로 또는 공유 스토리지 경로를 전달하라. |

### `prosody_input` 필수 키

`build_backend_payload()`가 런타임에 주입하는 키가 포함된 상태여야 한다.

| 키 | 타입 | 출처 | 설명 |
|---|---|---|---|
| `reference_text` | `str` | `backdata_export` | TTS 합성 및 forced alignment에 사용할 한국어 텍스트 |
| `audio_file_path` | `str` | `build_backend_payload()` 런타임 주입 | 학습자 WAV 파일 절대 경로 |
| `phoneme_segments` | `list[dict]` | `backdata_export` | 학습자 forced alignment 결과 (음소별 시간 경계) |

> **주의:** `audio_file_path`는 artifact JSON 파일에 저장되지 않는다.
> `build_backend_payload()` 호출 시 메모리에만 주입된다.
> 저장된 JSON을 나중에 다시 읽어 분석할 경우 직접 경로를 복원해야 한다.

---

## 반환값 — `list[dict]`

음절 수 = `min(native 음절 수, learner 음절 수)`.
모든 float 값은 JSON 직렬화 가능 (NaN은 `None`으로 변환됨).

### 필드 목록

| 필드 | 타입 | 설명 |
|---|---|---|
| `syllable_idx` | `int` | 음절 인덱스 (0부터) |
| `syllable_label` | `str` | 해당 음절에 대응하는 한글 문자. `reference_text`에서 공백·구두점을 제거한 순서. |
| `native_start` | `float` | 원어민 오디오 내 음절 시작 시각 (초). 시간축 시각화 시 x축 기준점으로 사용. |
| `learner_start` | `float` | 학습자 오디오 내 음절 시작 시각 (초). 시간축 시각화 시 x축 기준점으로 사용. |
| `native_f0` | `list[float]` | 원어민 TTS의 z-score 정규화 F0, 50프레임. 무성 구간은 `0.0` |
| `learner_f0` | `list[float]` | 학습자의 z-score 정규화 F0, 50프레임. 무성 구간은 `0.0` |
| `joint_voiced_mask` | `list[bool]` | 두 화자 모두 유성인 프레임. `False` 구간은 비교 불가 (아래 참고) |
| `native_duration` | `float` | 원어민 음절 지속시간 (초) |
| `learner_duration` | `float` | 학습자 음절 지속시간 (초) |
| `rmse` | `float \| None` | F0 RMSE (z-score 단위). 유성 프레임이 하나도 없으면 `None` |
| `pearson` | `float \| None` | 억양 흐름 유사도 (-1 ~ 1). 유성 프레임 < 5이면 `None` |
| `slope_diff` | `float \| None` | 피치 변화율 차이 (native - learner). 유성 프레임 < 3이면 `None` |
| `voiced_frame_count` | `int` | 유효 유성 프레임 수 |
| `duration_ratio` | `float \| None` | `learner_duration / native_duration`. native가 0이면 `None` |

### 메트릭 해석

| 메트릭 | 해석 | 권장 threshold |
|---|---|---|
| `rmse` | 값이 클수록 음높이 이탈 심각. z-score 단위이므로 화자 간 비교 가능 | 실험 필요 (현재 `plotter.py`는 `threshold=1` 사용) |
| `pearson` | +1에 가까울수록 억양 방향(상승·하강)이 일치. 0 이하면 역방향 | `< 0.5` → 경고 권장 |
| `slope_diff` | 양수: 학습자가 native보다 피치를 덜 올림/더 내림. 음수: 반대 | 절댓값 기준 판단 |
| `duration_ratio` | 1.0이 기준. 2.0이면 학습자가 2배 길게 발음 (러시아어 강세 이식 신호) | `> 1.5` 또는 `< 0.6` → 경고 권장 |

### 메트릭 → 피드백 매핑

메트릭은 조합해서 해석할 때 더 구체적인 피드백이 가능하다.

| 조건 | 진단 | 피드백 예시 |
|---|---|---|
| `rmse` 높음 | 음높이 이탈 | "이 음절의 음높이가 원어민과 크게 다릅니다" |
| `pearson` 낮음 (< 0.5) | 억양 방향 불일치 | "올라가야 할 곳에서 내려가거나, 그 반대입니다" |
| `pearson` 음수 | 억양 완전 역방향 | "억양 방향이 반대입니다. 원어민 발음을 다시 들어보세요" |
| `slope_diff` 양수 (큰 값) | 피치 변화 부족 | "이 음절에서 억양 변화가 너무 평탄합니다" |
| `slope_diff` 음수 (큰 값) | 피치 변화 과도 | "이 음절에서 억양 변화가 너무 급격합니다" |
| `duration_ratio` > 1.5 | 음절 과도 연장 | "이 음절을 너무 길게 발음했습니다" |
| `duration_ratio` < 0.6 | 음절 과도 단축 | "이 음절을 너무 짧게 발음했습니다" |
| `rmse` 높음 + `duration_ratio` > 1.5 | L1 transfer 의심 | "러시아어 강세 패턴이 이 음절에 영향을 주고 있을 수 있습니다" |
| `voiced_frame_count` == 0 | 무성음 처리 | 해당 음절은 메트릭 산출 불가. 피드백 생략 또는 별도 처리 필요 |

---

## `joint_voiced_mask` 시각화 활용

`joint_voiced_mask`는 두 화자 모두 유성음인 프레임을 나타내는 50개짜리 bool 배열이다.
`False` 구간은 한쪽 또는 양쪽이 무성음이어서 F0 비교가 의미 없는 구간이며, 메트릭도 이 mask 기준으로 계산된다.

```
True  구간: F0 곡선 실선, 불투명 → 신뢰 가능한 비교
False 구간: F0 곡선 점선 또는 반투명 → "비교 불가 구간" 시각적 표시
```

`False` 구간에서 `native_f0` / `learner_f0` 값은 `0.0`으로 채워져 있으므로,
무성 구간을 0으로 표시하거나 선을 끊어서 표현할 수 있다.

---

## 비동기 처리

`analyze()`는 동기 함수다. 내부적으로:

- Wav2Vec2 CTC forced alignment (CPU 연산, ~1~3초)
- Praat F0 추출 (경량, ~0.1초)

비동기 처리(`asyncio.to_thread`, Celery 등)는 API 설계자가 결정한다.
분석 시간이 긴 만큼 백그라운드 태스크로 실행하고 결과를 폴링하거나
웹소켓으로 스트리밍하는 방식을 권장한다.

---

## 성능 — 모델 로딩 및 캐시

### Wav2Vec2 싱글톤

`analyze()`는 내부적으로 `get_default_recognizer()`를 호출하며,
이 함수는 `@lru_cache(maxsize=1)`로 감싸져 있어 프로세스 내에서 모델을 한 번만 로드한다.

팀원 파이프라인과 같은 프로세스에서 실행하면 **모델을 공유하므로 중복 로드 없음**.
`recognizer` 파라미터를 `None`으로 두는 것이 기본 권장이다.

### TTS 캐시

동일한 `(text, voice, speed)` 조합은 SHA256 해시 기반으로 WAV 파일을 캐싱한다.
같은 문장에 대한 두 번째 호출부터는 Google Cloud API를 호출하지 않는다.

```python
results = analyze(prosody_input, tts_cache_dir="/shared/tts_cache")
```

---

## 프로덕션 운영 시 주의사항

### TTS 비용 및 교체 가능성

Google Cloud TTS Neural2 음성은 **유료**다. 캐시 히트 시에는 API를 호출하지 않으므로, 동일 문장 반복 요청에는 추가 비용이 없다.

- 신규 문장(캐시 미스) 1건당 API 호출 1회 발생
- 대량의 신규 문장이 단기간에 몰리면 비용이 급증할 수 있음
- [Google Cloud TTS 요금표](https://cloud.google.com/text-to-speech/pricing) 참고

**TTS 교체 가능:** `core/tts.py`의 `generate_tts()` 함수만 교체하면 다른 TTS 엔진으로 전환할 수 있다.
인터페이스는 `(text: str, cache_dir: Path, ...) -> Path` 형태만 유지하면 된다.

한국어를 지원하는 오픈소스 대안:

| 모델 | 특징 |
|---|---|
| [Coqui TTS](https://github.com/coqui-ai/TTS) | 다국어, HuggingFace 모델 허브에 한국어 모델 다수 |
| [MeloTTS](https://github.com/myshell-ai/MeloTTS) | 경량, 다국어(한국어 포함), 로컬 실행 |
| [VITS / VITS2](https://github.com/jaywalnut310/vits) | 고품질, HuggingFace에 한국어 파인튜닝 모델 존재 |
| [Piper](https://github.com/rhasspy/piper) | 빠른 로컬 추론, 경량 배포에 적합 |

### 캐시 용량 관리

WAV 파일은 별도 만료 정책 없이 누적된다. 학습 콘텐츠 수가 고정적이면 자연스럽게 수렴하지만,
동적으로 새로운 문장이 계속 추가되는 구조라면 주기적 cleanup 또는 LRU 캐시 적용이 필요하다.

### 동시 요청 시 캐시 race condition

현재 구현은 파일 락이 없어 멀티 프로세스/컨테이너 환경에서 같은 캐시 키에 쓰기 충돌이 발생할 수 있다.

- **단일 프로세스 환경**: 문제 없음
- **멀티 프로세스/컨테이너 환경**: 파일 수준 락(`fcntl.flock`) 또는 원자적 쓰기(`tmpfile → os.replace`) 적용 필요

캐시를 외부 스토리지(S3, GCS 등)로 대체하면 이 문제와 용량 관리를 함께 해결할 수 있다.

---

## 의존성 및 설정

### Google Cloud TTS 인증 (최초 1회 설정)

**0단계 — Google Cloud CLI 설치**

`gcloud` 명령어를 사용하려면 [Google Cloud CLI](https://cloud.google.com/sdk/docs/install?hl=ko)가 필요하다.

```bash
gcloud init
```

**1단계 — 프로젝트 생성 및 결제 계정 연결**

- [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트를 생성하거나 선택한다.
- `결제(Billing)` 메뉴에서 결제 계정이 연결되어 있는지 확인한다.
  Cloud TTS는 **결제 수단이 연결된 프로젝트에서만 API를 활성화할 수 있다.**

**2단계 — Text-to-Speech API 활성화**

```bash
gcloud services enable texttospeech.googleapis.com --project=[PROJECT_ID]
```

**3단계 — 인증 설정**

```bash
# 개발 환경
gcloud auth application-default login

# 프로덕션 환경 (서비스 계정 권장)
# Console → API 및 서비스 → 사용자 인증 정보 → 서비스 계정 생성
# → roles/cloudtexttospeech.user 권한 부여 → JSON 키 다운로드
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> JPype1 빌드 시 `ant`가 필요하다:
> ```bash
> brew install ant  # macOS
> ```

---

## 전체 호출 예시

```python
from pronunciation_backend_pipeline import evaluate_pronunciation_file
from analyze import analyze

response = evaluate_pronunciation_file(
    audio_path="recordings/user_sample.wav",
    reference_text="오늘 날씨가 정말 좋네요",
)

if response["status"]["evaluation_status"] != "ready":
    print("게이트 탈락:", response["status"]["status_message"])
else:
    results = analyze(response["prosody_input"])

    for syllable in results:
        print(
            f"음절 {syllable['syllable_idx']}: "
            f"RMSE={syllable['rmse']}, "
            f"Pearson={syllable['pearson']}, "
            f"duration_ratio={syllable['duration_ratio']}"
        )
```

### 출력 예시 (JSON)

```json
[
  {
    "syllable_idx": 0,
    "native_f0": [0.12, 0.34, "..."],
    "learner_f0": [-0.05, 0.21, "..."],
    "joint_voiced_mask": [true, true, false, "..."],
    "native_duration": 0.18,
    "learner_duration": 0.31,
    "rmse": 0.82,
    "pearson": 0.71,
    "slope_diff": -0.15,
    "voiced_frame_count": 38,
    "duration_ratio": 1.72
  }
]
```

---

## 시각화 Use Case

`native_f0`와 `learner_f0`를 하나의 그래프에 겹쳐 그리고, 음절 경계마다 레이블과 메트릭을 표시하는 구성을 권장한다.

- **X축**: `native_start`를 기준점으로, `native_duration`에 걸쳐 50프레임을 선형 매핑
- **Y축**: z-score 정규화 F0 값
- **두 곡선**: native(파랑) / learner(빨강) 겹쳐 표시
- **voiced 구분**: `joint_voiced_mask == False` 구간은 점선 또는 반투명 처리
- **음절 경계**: `native_start`마다 수직선 표시
- **음절 주석**: 경계 중앙에 `syllable_label` + `rmse` + `pearson` 표시. `rmse`가 threshold 초과 시 빨간색 강조

---

---

# Appendix — 배경 설명

API 동작 이해에 도움이 되는 설계 배경이다. 호출에는 필요하지 않다.

---

## 핵심 용어

| 용어 | 설명 |
|---|---|
| **F0 (Fundamental Frequency)** | 음의 높이(pitch). 성대가 1초에 몇 번 진동하는지를 나타내는 주파수(Hz). 억양 분석의 핵심 신호. |
| **유성음 (voiced)** | 성대가 진동하며 F0가 존재하는 구간. 모음과 일부 자음이 해당. |
| **무성음 (unvoiced)** | 성대가 진동하지 않아 F0가 없는 구간. 'ㅅ', 'ㅍ' 등의 자음. |
| **z-score 정규화** | 화자마다 다른 음역대(남성/여성 등)를 제거하고 상대적인 패턴만 비교하기 위한 정규화. |
| **음절 경계** | 각 음절이 시작되고 끝나는 시간 구간. forced alignment로 추출. |
| **Segmental Alignment** | 음절 인덱스를 1:1로 대응시켜 비교하는 방식. native의 1번 음절 ↔ learner의 1번 음절. |

---

## 내부 동작 방식

`prosody_input`에는 학습자 정보만 담겨 있어 억양 비교를 위한 원어민 기준값이 없다.
이를 보완하기 위해 아래 순서로 파이프라인이 실행된다.

```
1. reference_text → Google Cloud TTS → native WAV 합성
2. native WAV → Wav2Vec2 forced alignment → 음소별 시간 경계
3. native / learner 각각 Praat으로 F0 추출 → z-score 정규화
4. 음소 경계 → 음절 경계로 변환
5. Segmental Alignment: 음절 인덱스 1:1 매핑, 각 음절을 50프레임으로 리샘플
   → 특정 음절에 대해 native / learner의 pitch를 같은 선 상에서 비교 가능
6. 음절별 멀티 메트릭 계산 → JSON 직렬화 → 반환
```

**왜 TTS를 쓰는가?**
원어민 기준 오디오가 별도로 제공되지 않으므로 TTS로 합성한다.
동일 텍스트에 대해 일관된 발화 품질을 보장하며, F0 추출 안정성이 높다.

**왜 forced alignment를 native에도 실행하는가?**
음절별 비교를 하려면 양쪽의 음소 경계가 필요하다.
학습자 경계는 팀원 파이프라인이 이미 제공하지만, native(TTS) 경계는 직접 추출해야 한다.
이 때문에 `_forced_align()`은 `core/`가 아닌 `src/`(음소분석 모듈)에 의존한다.

**왜 DTW가 아닌 Segmental Alignment인가?**
DTW는 F0 패턴이 유사하다는 전제로 시간축을 정렬한다.
억양 오류가 있는 학습자 발화에서는 이 전제가 깨지므로 정렬 오류와 억양 오류를 구분할 수 없다.
Segmental Alignment는 음소 위치 기준으로 1:1 대응하므로 억양이 달라도 비교 구조가 보장된다.