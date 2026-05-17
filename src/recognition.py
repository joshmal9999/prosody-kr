from __future__ import annotations

"""Richer audio-recognition wrapper for alignment-driven evaluation."""

from pathlib import Path

import numpy as np
import torch

from src.audio_to_ipa import AudioToIPARecognizer
from src.label_to_ipa import decode_label_text
from src.quality import analyze_audio_quality
from src.types import AudioRecognitionResult


def _trim_audio_for_recognition(audio: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, float]:
    frame_size = max(1, int(sampling_rate * 0.02))
    padding = max(1, int(sampling_rate * 0.15))
    frame_energy = [
        float(np.sqrt(np.mean(np.square(audio[index:index + frame_size]))))
        for index in range(0, len(audio), frame_size)
        if len(audio[index:index + frame_size])
    ]
    if not frame_energy:
        return audio, 0.0

    threshold = max(0.01, 0.08 * max(frame_energy))
    active_frames = [index for index, energy in enumerate(frame_energy) if energy >= threshold]
    if not active_frames:
        return audio, 0.0

    start_sample = max(0, active_frames[0] * frame_size - padding)
    end_sample = min(len(audio), (active_frames[-1] + 1) * frame_size + padding)
    if end_sample <= start_sample:
        return audio, 0.0
    return audio[start_sample:end_sample], start_sample / float(sampling_rate)


def recognize_audio(recognizer: AudioToIPARecognizer, audio_path: str | Path) -> AudioRecognitionResult:
    """Run the recognizer and keep metadata needed for later constrained decoding."""

    loaded = recognizer.load_audio(audio_path, target_sr=16000)
    if isinstance(loaded, tuple):
        audio, sampling_rate = loaded
    else:
        audio, sampling_rate = loaded, 16000
    quality_report = analyze_audio_quality(audio, sampling_rate)
    alignment_audio, alignment_offset = _trim_audio_for_recognition(audio, sampling_rate)
    inputs = recognizer.processor(alignment_audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(recognizer.device)
    attention_mask = inputs.attention_mask.to(recognizer.device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits_tensor = recognizer.model(input_values=input_values, attention_mask=attention_mask).logits

    pred_ids = torch.argmax(logits_tensor, dim=-1)
    raw_label_text = recognizer.processor.batch_decode(pred_ids)[0]
    raw_label_text = raw_label_text.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
    raw_labels, sequence = decode_label_text(raw_label_text)

    logits = logits_tensor[0].detach().cpu()
    frame_confidence = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()
    duration = len(alignment_audio) / float(sampling_rate)
    # Store frame boundaries, not centers. There are N logit frames and N+1
    # boundaries, so the last segment can end exactly at the audio duration.
    frame_count = len(frame_confidence)
    frame_timestamps = [
        alignment_offset + duration * index / max(1, frame_count)
        for index in range(frame_count + 1)
    ]

    return AudioRecognitionResult(
        raw_text=sequence.raw_text,
        normalized_text=sequence.normalized_text,
        tokens=sequence.tokens,
        raw_label_text=raw_label_text,
        raw_labels=raw_labels,
        logits=logits.tolist(),
        frame_confidence=frame_confidence,
        frame_timestamps=frame_timestamps,
        sampling_rate=sampling_rate,
        quality_report=quality_report,
    )