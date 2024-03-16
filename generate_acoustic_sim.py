import json
import math
import os
import sys

import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm

from lib.room import custom_plot
from src.class_room import Room
from src.class_sound import Ambient, AudioLoader, Drone, Voice
from src.file_io import load_config, write_signal_to_wav
from src.snr import adjust_snr

pra.Room.plot = custom_plot


def calculate_angles(positions, mic_center):
    angles = []
    for position in positions:
        dx = position[0] - mic_center[0]
        dy = position[1] - mic_center[1]
        angles.append(math.atan2(dy, dx))
    return sorted(angles)


def export_ans(mic_center, output_dir, voice, ambient):
    ans = {}
    if voice is not None:
        voice_ans = calculate_angles(voice.positions, mic_center)
        ans["voice"] = voice_ans
    if ambient is not None:
        ambient_ans = calculate_angles(ambient.positions, mic_center)
        ans["ambient"] = ambient_ans

    with open(f"{output_dir}/ans.json", "w") as f:
        json.dump(ans, f, indent=4)


def main(config, output_dir):
    room = Room(config["pra"], config["seed"])
    AudioLoader.initialize_x_positions_pool(room)
    voice = Voice(config["voice"], config["n_voice"], config["seed"], fs=room.fs, room=room)
    drone = Drone(config["drone"], config["seed"], fs=room.fs)
    if config["n_ambient"] != 0:
        ambient = Ambient(config["ambient"], config["n_ambient"], config["seed"], fs=room.fs, room=room)
    else:
        ambient = None

    room.place_microphones(drone.mic_positions)

    adjust_snr(room, voice, drone, drone.snr, output_dir)
    if ambient is not None:
        adjust_snr(room, voice, ambient, ambient.snr, output_dir)

    room.place_source(voice=voice, drone=drone, ambient=ambient)

    start = int(room.fs * config["processing"]["start_time"])
    end = int(room.fs * config["processing"]["end_time"])

    simulated_signals = room.simulate(output_dir)
    max_val = max(np.max(np.abs(simulated_signals[name])) for name in simulated_signals)

    for name in simulated_signals:
        signal = simulated_signals[name][:, start:end]
        write_signal_to_wav(signal, f"{output_dir}/{name}.wav", room.fs, max_val)

    export_ans((0, config["drone"]["mic_positions"]["height"]), output_dir, voice, ambient)


def update_config(
    config, height, roughness, material, n_voice, n_ambient, snr_ego, snr_ambient
):
    config["drone"]["mic_positions"]["height"] = height
    config["pra"]["room"]["floor"]["roughness"] = roughness
    config["pra"]["room"]["floor"]["material"] = material
    config["n_voice"] = n_voice
    config["n_ambient"] = n_ambient
    config["drone"]["snr"] = snr_ego
    config["ambient"]["snr"] = snr_ambient
    return config


def create_output_directory(*args):
    output_dir = f"experiments/data/{';'.join(map(str, args))}/simulation"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


class TqdmPrintRedirect:
    def __init__(self, tqdm_object):
        self.tqdm_object = tqdm_object
        self.original_write = tqdm_object.fp.write

    def write(self, message):
        # tqdmのオリジナルwriteメソッドを使用
        self.original_write(message)

    def flush(self):
        pass


def safe_main(config, output_dir, attempt=1):
    try:
        main(config, output_dir)
    except Exception as e:
        if attempt < 3:
            print(f"Error occurred, retrying... Attempt {attempt}")
            safe_main(config, output_dir, attempt + 1)
        else:
            print(f"Failed after 3 attempts. Error: {e}")
            raise


if __name__ == "__main__":
    config = load_config("experiments/config.yaml")
    np.random.seed(config["seed"])

    heights = [2, 3, 4, 5]
    roughnesses = [[0.1, 1.0], [0.2, 1.2]]
    materials = ["hard_surface", "plasterboard", "wooden_lining"]
    n_voices = [1, 2, 3]
    n_ambients = [0, 1, 2]
    snr_egos = [8, 11, 14]
    snr_ambients = [-3, 0, 3]

    params_list = [
        (height, roughness, material, n_voice, n_ambient, snr_ego, snr_ambient)
        for height in heights
        for roughness in roughnesses
        for material in materials
        for n_voice in n_voices
        for n_ambient in n_ambients
        for snr_ego in snr_egos
        for snr_ambient in ([0] if n_ambient == 0 else snr_ambients)
    ]

    # プログレスバーがアクティブな間、sys.stdoutを上書き
    with tqdm(total=len(params_list)) as pbar:
        original_stdout = sys.stdout
        sys.stdout = TqdmPrintRedirect(pbar)

        for params in params_list:
            updated_config = update_config(config, *params)
            output_dir = create_output_directory(*params)
            safe_main(config, output_dir)
            pbar.update(1)

        # プログレスバー終了後に元のstdoutに戻す
        sys.stdout = original_stdout
