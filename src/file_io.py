import os
import wave
from typing import Tuple

import numpy as np
import yaml
from pydub import AudioSegment
from scipy.io import wavfile
from scipy.signal import resample_poly

from .audio_processing import scale_signal


def convert_to_wav(audio_file_path: str) -> str:
    file_ext = os.path.splitext(audio_file_path)[1][1:]
    audio = AudioSegment.from_file(audio_file_path, file_ext)
    wav_file_path = os.path.splitext(audio_file_path)[0] + ".wav"
    audio.export(wav_file_path, format="wav")
    return wav_file_path


def ensure_dir(file_path: str) -> None:
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def write_signal_to_wav(signal: np.ndarray, wav_file_path: str, sample_rate: int, max_val: float = -1) -> None:
    ensure_dir(wav_file_path)

    signal = scale_signal(signal, max_val)

    if len(signal.shape) == 1:
        channels = 1
    else:
        channels = signal.shape[0]
        signal = signal.T.flatten()

    with wave.open(wav_file_path, "w") as wave_out:
        wave_out.setnchannels(channels)
        wave_out.setsampwidth(2)
        wave_out.setframerate(sample_rate)
        wave_out.writeframes(signal.tobytes())


def write_signal_to_npz(signal: np.ndarray, npz_file_path: str, sample_rate: int) -> None:
    ensure_dir(npz_file_path)
    np.savez(npz_file_path, signal=signal, sample_rate=sample_rate)


def load_signal_from_npz(npz_file_path: str) -> Tuple[np.ndarray, int]:
    data = np.load(npz_file_path)
    return data["signal"], data["sample_rate"]


def load_signal_from_wav(wav_file_path: str, expected_fs: int) -> np.ndarray:
    fs, signal = wavfile.read(wav_file_path)
    if fs != expected_fs:
        signal = signal.astype(float)
        signal = resample_poly(signal, expected_fs, fs)

    if len(signal.shape) >= 2:
        signal = signal.T
    return signal


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
