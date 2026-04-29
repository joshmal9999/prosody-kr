import matplotlib.pyplot as plt
import numpy as np
from core.comparator import ComparisonResult
from core.f0_extractor import extract_f0
from core.metrics import SyllableRMSE

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


class ComparisonPlotter:
    NATIVE_COLOR = '#185FA5'
    LEARNER_COLOR = '#993C1D'

    def __init__(
        self,
        native_label: str = '원어민',
        learner_label: str = '학습자',
        threshold: float = 0.5,
    ):
        self.native_label = native_label
        self.learner_label = learner_label
        self.threshold = threshold

    def plot(
        self,
        result: ComparisonResult,
        syllable_rmse: list[SyllableRMSE],
        title: str = '억양비교',
        syllable_labels: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 4))
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)

        self._draw_f0_lines(ax, result)
        self._draw_syllable_annotations(ax, result, syllable_rmse, syllable_labels)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel('프레임')
        ax.set_ylabel('음높이 (z-score)')
        ax.set_ylim(-3.5, 3)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        plt.show()

    def _draw_f0_lines(self, ax, result: ComparisonResult) -> None:
        frames = np.arange(len(result.aligned_native))
        for f0, label, color in [
            (result.aligned_native, self.native_label, self.NATIVE_COLOR),
            (result.aligned_learner, self.learner_label, self.LEARNER_COLOR),
        ]:
            ax.plot(frames, f0, color=color, linewidth=2.5, label=label)

    def _draw_syllable_annotations(
        self,
        ax,
        result: ComparisonResult,
        syllable_rmse: list[SyllableRMSE],
        syllable_labels: list[str] | None = None,
    ) -> None:
        learner_times_aligned = result.learner_times[result.learner_indices]
        for syl in syllable_rmse:
            in_syl = np.where(
                (learner_times_aligned >= syl.t_start) & (learner_times_aligned < syl.t_end)
            )[0]
            if len(in_syl) == 0:
                continue
            ax.axvline(x=in_syl[0], color='gray', linestyle='--', linewidth=0.5)
            mid = int(in_syl.mean())
            if np.isnan(syl.rmse):
                text, color = 'N/A', 'gray'
            elif syl.rmse > self.threshold:
                text, color = f'{syl.rmse:.2f}', 'red'
                print(f"음절 {syl.syllable_idx}: {syl.t_start:.2f}s ~ {syl.t_end:.2f}s  (RMSE={syl.rmse:.3f})")
            else:
                text, color = f'{syl.rmse:.2f}', 'gray'
            ax.text(mid, 2.7, text, ha='center', fontsize=8, color=color)
            if syllable_labels and syl.syllable_idx < len(syllable_labels):
                ax.text(mid, -3.2, syllable_labels[syl.syllable_idx], ha='center', fontsize=10, color=color)

    def plot_raw_f0(
        self,
        native_path: str,
        learner_path: str,
        title: str = 'Normalized Pitch 비교 (Hz)',
        save_path: str | None = None,
    ) -> None:
        """DTW 정렬 없이 각 wav의 실제 시간축 F0(Hz)를 나란히 그린다."""
        native_f0 = extract_f0(native_path)
        learner_f0 = extract_f0(learner_path)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
        fig.suptitle(title, fontsize=14)

        for ax, f0_result, label, color in [
            (ax1, native_f0, self.native_label, self.NATIVE_COLOR),
            (ax2, learner_f0, self.learner_label, self.LEARNER_COLOR),
        ]:
            ax.plot(f0_result.times, f0_result.f0, color=color, linewidth=2, label=label)
            ax.set_ylabel('F0 (Normalized Hz)')
            ax.set_xlabel('시간 (초)')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()