from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from src.class_room import Room
from src.class_sound import Ambient, Drone, Voice
from src.file_io import write_signal_to_wav


def calculate_power(signal: np.ndarray) -> float:
    # signalの型がfloatであることを確認
    # int16などだとオーバーフローする可能性がある
    assert signal.dtype == float
    return np.sum(signal ** 2) / len(signal)


def calculate_snr(signal_s: np.ndarray, signal_n: np.ndarray,
                  n_data_s: int, n_data_n: int) -> float:
    power_s = calculate_power(signal_s) / n_data_s
    power_n = calculate_power(signal_n) / n_data_n
    snr = 10 * np.log10(power_s / power_n)
    return snr


def calculate_coef(signal_s: np.ndarray, signal_n: np.ndarray,
                   n_data_s: int, n_data_n: int, snr_target: float) -> float:
    power_s = calculate_power(signal_s) / n_data_s
    power_n = calculate_power(signal_n) / n_data_n
    # logの真数
    argument = 10 ** (snr_target / 10)
    # 係数の計算
    coefficient = np.sqrt(argument / (power_s / power_n))
    return coefficient


def get_sn_rec(room: Room, source: Voice,
               noise: Union[Drone, Ambient], mic_loc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    test_room_s = pra.Room.from_corners(room.corners, fs=room.fs, max_order=room.max_order)
    test_room_n = pra.Room.from_corners(room.corners, fs=room.fs, max_order=0)
    for test_room, signal_data in zip([test_room_s, test_room_n], [source, noise]):
        for signal, position in zip(signal_data.source, signal_data.positions):
            test_room.add_source(position, signal=signal)
        test_room.add_microphone_array(pra.MicrophoneArray(mic_loc, room.fs))
        test_room.simulate()
    return test_room_s.mic_array.signals[0], test_room_n.mic_array.signals[0]


def confirm_rec(room: Room, source: Voice,
                noise: Union[Drone, Ambient], mic_loc: np.ndarray, filename: str) -> None:
    test_room = pra.Room.from_corners(room.corners, fs=room.fs, max_order=room.max_order)
    for signal, position in zip(source.source, source.positions):
        test_room.add_source(position, signal=signal)
    for signal, position in zip(noise.source, noise.positions):
        test_room.add_source(position, signal=signal)
    test_room.add_microphone_array(pra.MicrophoneArray(mic_loc, room.fs))
    test_room.plot()
    plt.savefig(f"{filename}.png")
    test_room.simulate()
    write_signal_to_wav(test_room.mic_array.signals, f"{filename}.wav", room.fs)


def adjust_snr(room: Room, source: Voice,
               noise: Union[Drone, Ambient], snr_target: float, output_dir: str) -> None:
    """雑音をSNRに合わせて調整する"""
    mic_loc = room.rooms["source"].mic_array.center
    rec_s, rec_n = get_sn_rec(room, source, noise, mic_loc)
    snr = calculate_snr(rec_s, rec_n, source.n_sound, noise.n_sound)
    print(f"SNR before adjustment: {snr:.3f}")
    confirm_rec(room, source, noise, mic_loc, f"{output_dir}/before")

    coef = calculate_coef(rec_s, rec_n, source.n_sound, noise.n_sound, snr_target)
    print(f"Adjustment coefficient: {coef}")
    source_adjusted = []
    for signal in noise.source:
        source_adjusted.append((signal / coef))
    noise.source = source_adjusted

    rec_s, rec_n = get_sn_rec(room, source, noise, mic_loc)
    snr = calculate_snr(rec_s, rec_n, source.n_sound, noise.n_sound)
    print(f"SNR after adjustment: {snr:.3f}")
    confirm_rec(room, source, noise, mic_loc, f"{output_dir}/after")
