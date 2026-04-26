# Evaluation JSON Guide

이 문서는 `artifacts/` 아래에 저장되는 평가 산출물과 JSON 필드를 설명합니다.

## 생성 목적

- 발음 피드백용 LLM 입력 데이터 제공
- 억양/운율 분석용 음소 시간 구간 데이터 제공
- 평가 상태, gate 결과, alignment 결과를 한 파일에 함께 보존

## 저장 구조

평가 1회마다 `artifacts/` 바로 아래에 시각 기반 폴더 1개가 생성됩니다.

예시:

```text
artifacts/
└─ 20260421_214512_123456/
   ├─ 20260421_214512_123456.json
   └─ 20260421_214512_123456.wav
```

규칙은 다음과 같습니다.

- 폴더명은 실행 시각 기반 `YYYYMMDD_HHMMSS_microseconds`
- JSON 파일명은 폴더명과 동일한 basename 사용
- 음성 파일도 같은 basename을 사용하고 확장자만 원본을 유지
- 성공(`ready`), 재녹음 요청(`retry`), 결과 폐기(`discarded`) 모두 저장

## 평가 파이프라인 요약

현재 파이프라인은 아래 순서로 동작합니다.

1. 음성 품질 gate
2. token-level IPA 거친 정렬 gate
3. forced alignment 수행 후 alignment 신뢰도 gate
4. 3단계까지 통과하면 점수 계산과 오류 분석 수행

JSON의 `gates`에는 1, 2, 3번 gate의 `performed`와 `passed`가 모두 기록됩니다.

## Top-level 필드

### `schema_version`
- JSON 포맷 버전

### `created_at`
- 파일 생성 시각(로컬 타임존 포함 ISO format)

### `profile`
- 평가에 사용한 학습자 프로필

### `audio_source_name`
- 업로드 파일명 또는 마이크 입력 기본 이름

### `artifact_bundle`
- 현재 산출물 묶음 정보
- `artifact_id`: 시각 폴더 이름
- `json_file_name`: 저장된 JSON 파일명
- `audio_file_name`: 함께 저장된 음성 파일명

### `status`
- `evaluation_status`: `ready` / `retry` / `discarded`
- `status_message`: 현재 상태 설명

### `reference`
- 정답 문장, 대표 발음형, 대표 IPA, 선택된 후보, 전체 후보 목록

### `recognition`
- 사용자 추정 IPA
- raw label text / raw labels
- 사용자 IPA 토큰 목록

### `gates`
- 현재 파이프라인의 gate 통과 여부 요약

구성:

- `audio_quality_gate`
  - `performed`
  - `passed`
  - `report`
- `coarse_token_alignment_gate`
  - `performed`
  - `passed`
  - `normalized_score`
- `alignment_confidence_gate`
  - `performed`
  - `passed`
  - `report`

### `alignment`
- `coarse`: token-level weighted alignment 결과
- `forced`: frame-level forced alignment 결과

### `evaluation`
- score breakdown
- feedback report

### `llm_feedback_input`
- 발음 피드백 담당 LLM이 바로 사용하기 좋은 요약 입력
- 주요 필드:
  - `reference_text`
  - `reference_pronunciation`
  - `reference_ipa`
  - `user_ipa`
  - `status`
  - `status_message`
  - `score_breakdown`
  - `gate_summary`
  - `mismatches`
  - `issues`

### `prosody_input`
- 억양/운율 분석 담당자가 바로 사용하기 좋은 입력
- 주요 필드:
  - `reference_text`
  - `selected_reference_pronunciation`
  - `selected_reference_ipa`
  - `gate_summary`
  - `phoneme_segments`
  - `alignment_confidence`

### `debug`
- 내부 디버그용 정보

## `phoneme_segments` 설명

`prosody_input.phoneme_segments`는 음소별 시간 구간 정보입니다.

각 항목은 다음 필드를 가집니다.

- `token`: IPA 음소
- `label`: 모델 label space 심벌
- `start_time`: 시작 시각(초)
- `end_time`: 끝 시각(초)
- `duration`: 길이(초)
- `frame_start`: 시작 프레임 index
- `frame_end`: 끝 프레임 index
- `confidence`: 해당 음소 구간 평균 confidence

예시:

```json
{
  "token": "tɕ",
  "label": "J",
  "start_time": 0.82,
  "end_time": 0.94,
  "duration": 0.12,
  "frame_start": 41,
  "frame_end": 46,
  "confidence": 0.73
}
```

## 활용 권장 방식

### 발음 피드백 LLM
- `llm_feedback_input`를 우선 사용
- 필요 시 `gates`, `alignment.coarse`, `evaluation.feedback_report`를 추가 참고

### 억양/운율 분석
- `prosody_input.phoneme_segments`를 기본 입력으로 사용
- `prosody_input.alignment_confidence`와 `gates.alignment_confidence_gate`를 함께 보고 신뢰도 판단

## 전달용 짧은 설명문

아래 문구를 그대로 공유해도 됩니다.

```text
평가 결과는 추론 1회마다 `artifacts/<시각>/` 폴더로 저장됩니다.
각 폴더 안에는 같은 basename의 JSON과 원본 음성 파일이 같이 들어 있습니다.

`llm_feedback_input`은 발음 피드백 생성용 입력이고,
`prosody_input`은 음소별 시작/끝 시각 기반 억양 분석용 입력입니다.
특히 `prosody_input.phoneme_segments`에 각 음소의 start_time, end_time, duration, confidence가 들어 있습니다.

또한 `gates`에 아래 3단계의 통과 여부가 기록됩니다.
1. 음성 품질 gate
2. token-level 거친 정렬 gate
3. alignment 신뢰도 gate

그래서 JSON만 봐도 이번 결과가 정상 평가인지, 재녹음 권장인지, alignment 신뢰도 부족으로 폐기된 것인지 바로 판단할 수 있습니다.
```
