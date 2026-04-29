import os
import io
import numpy as np
import pandas as pd
import soundfile as sf

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")


def to_pcm16_mono(path: str, index: int = 0) -> str:
    df = pd.read_parquet(path)
    audio_bytes = df["audio"][index]["bytes"]
    data, samplerate = sf.read(io.BytesIO(audio_bytes))

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = (data * 32767).clip(-32768, 32767).astype(np.int16)

    output_path = os.path.join(DATA_DIR, f"sample_{index}.wav")
    sf.write(output_path, data, samplerate, subtype="PCM_16")
    return output_path