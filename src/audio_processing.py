import os
import wave
from typing import List, Optional, Tuple

import numpy as np
from pydub import AudioSegment


def normalize_and_pad_audio_files(wav_file_paths: List[str]) -> List[np.ndarray]:
    audio_data = []
    n_samples = 0
    n_sources = len(wav_file_paths)

    for wav_file_path in wav_file_paths:
        with wave.open(wav_file_path) as wav:
            data = wav.readframes(wav.getnframes())
            data = np.frombuffer(data, dtype=np.int16)
            n_samples = max(wav.getnframes(), n_samples)
            data = data / np.iinfo(np.int16).max
            audio_data.append(data)

    for s in range(n_sources):
        if len(audio_data[s]) < n_samples:
            pad_width = n_samples - len(audio_data[s])
            audio_data[s] = np.pad(audio_data[s], (0, pad_width), "constant")
        audio_data[s] /= np.std(audio_data[s])
    return audio_data


def modify_audio_volume(input_wav: str, output_wav: Optional[str], volume_delta: float) -> None:
    sourceAudio = AudioSegment.from_wav(input_wav)
    processedAudio = sourceAudio + volume_delta
    if output_wav is None:
        basename = os.path.basename(input_wav)
        basename = os.path.splitext(basename)[0]
        output_wav = f"{basename}_{volume_delta}.wav"
        output_dir = os.path.dirname(input_wav).replace("raw", "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_wav = os.path.join(output_dir, output_wav)
    processedAudio.export(output_wav, format="wav")


def scale_signal(signal: np.ndarray) -> np.ndarray:
    # スケーリングファクターを信号の最大絶対値に設定
    scaling_factor = np.max(np.abs(signal))
    # 音声データをスケーリングして int16 に変換
    return (signal * np.iinfo(np.int16).max / scaling_factor).astype(np.int16)


def calculate_snr(desired: np.ndarray, out: np.ndarray) -> float:
    # 短い方の長さを取得
    min_length = min(desired.shape[0], out.shape[0])
    # 信号と出力を同じ長さにクリップ
    desired = desired[:min_length]
    out = out[:min_length]
    # 残余ノイズを計算
    residual_noise = desired - out
    # SNRを計算
    signal_power = np.sum(desired ** 2)
    noise_power = np.sum(residual_noise ** 2)

    if noise_power == 0:
        return float("inf")

    snr = 10.0 * np.log10(signal_power / noise_power)
    return snr


def detect_silent_intervals(fs: int, signal: np.ndarray) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    # Convert to single channel if multi-channel
    if len(signal.shape) != 1:
        signal = signal[:, 0]

    # Detect the end of the initial silence
    non_zero = np.where(signal != 0)
    if len(non_zero[0]) == 0:
        raise ValueError("The signal is all zeros")

    start_index = non_zero[0][0]
    end_index = non_zero[-1][-1] + 1
    start_time = start_index / fs
    end_time = end_index / fs

    return (start_index, end_index), (start_time, end_time)


def clip_signal(signal: np.ndarray, start_index: Optional[int] = None, end_index: Optional[int] = None,
                auto: bool = False, fs: int = 16000, buffer: int = 0) -> np.ndarray:
    if auto or start_index is None or end_index is None:
        (start_index, end_index), _ = detect_silent_intervals(fs, signal)
    return signal[start_index + buffer:end_index - buffer]
