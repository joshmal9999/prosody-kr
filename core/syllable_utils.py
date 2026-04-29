from __future__ import annotations

VOWEL_BASES = set("aeiouyɯɨʉɪʊøœɛɜɞəɐɔæɑɒʌ")


def _is_vowel(token: str) -> bool:
    return any(ch in VOWEL_BASES for ch in token)


def segments_to_syllable_boundaries(
    segments: list[dict],
) -> list[tuple[float, float]]:
    """음소 segments → 음절 단위 시간 경계 변환.

    각 음절 = onset consonant(s) + nucleus(vowel) 기준.
    nucleus 이후 coda consonants는 다음 음절 onset으로 처리.
    (coda는 무성음이라 voiced_mask에서 걸러지므로 귀속 방향이 결과에 영향 없음)

    Args:
        segments: 각 dict에 'token', 'start_time', 'end_time' 필요.

    Returns:
        list of (t_start, t_end) per syllable, nucleus 수만큼
    """
    nucleus_indices = [i for i, seg in enumerate(segments) if _is_vowel(seg["token"])]

    boundaries = []
    for k, nuc_idx in enumerate(nucleus_indices):
        prev_nuc_idx = nucleus_indices[k - 1] if k > 0 else -1
        onset_start_idx = prev_nuc_idx + 1
        t_start = segments[onset_start_idx]["start_time"]
        t_end = segments[nuc_idx]["end_time"]
        boundaries.append((t_start, t_end))

    return boundaries
