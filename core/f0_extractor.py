import parselmouth
import numpy as np
from dataclasses import dataclass


@dataclass
class F0Result:
    f0: np.ndarray        # z-score 정규화된 f0
    f0_raw: np.ndarray    # 원본 Hz 값
    voiced_mask: np.ndarray
    times: np.ndarray
    sampling_rate: float

    @property
    def voiced(self) -> np.ndarray:
        return self.f0[self.voiced_mask]

    @property
    def voiced_raw(self) -> np.ndarray:
        return self.f0_raw[self.voiced_mask]

    @property
    def voiced_count(self) -> int:
        return int(self.voiced_mask.sum())

    @property
    def total_frames(self) -> int:
        return len(self.f0)


def normalize_f0_zscore(f0):
    voiced = f0[f0 > 0]  # f0=0은 무성음(unvoiced)이라 제외
    mean = np.mean(voiced)  # 유성음 구간의 평균 pitch
    std = np.std(voiced)  # 유성음 구간의 표준편차

    f0_norm = np.zeros_like(f0)  # 0으로 채운 동일 크기 배열 생성
    f0_norm[f0 > 0] = (f0[f0 > 0] - mean) / std  # 유성음 구간만 normalize
    return f0_norm

def extract_f0(wav_path: str) -> F0Result:
    snd = parselmouth.Sound(wav_path)
    snd_resampled = snd.resample(new_frequency=16000, precision=50)
    pitch = snd_resampled.to_pitch()

    times = pitch.xs() # 각 frame의 시간(초)


    # frame당 f0
    f0_raw = pitch.selected_array['frequency']
    voiced_mask = f0_raw > 0
    f0 = normalize_f0_zscore(f0_raw)

    return F0Result(
        f0=f0,
        f0_raw=f0_raw,
        voiced_mask=voiced_mask,
        times=times,
        sampling_rate=snd.sampling_frequency,
    )