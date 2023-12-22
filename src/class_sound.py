import numpy as np
from glob import glob
import pyroomacoustics as pra

from src.file_io import load_signal_from_wav


class AudioLoader:
    def __init__(self, config, n_sound, fs=16000):
        self.n_sound = n_sound
        self.source = []
        source_dir = config["source_dir"]
        source_files = np.random.choice(
            glob(f"{source_dir}/*.wav"), n_sound, replace=False
        )
        for file_path in source_files:
            signal = load_signal_from_wav(file_path, fs)
            self.source.append(signal)


class Voice(AudioLoader):
    def __init__(self, config, n_sound, fs, room):
        super().__init__(config, n_sound, fs)

        room_x = np.append(room.corners[0][3:], room.corners[0][0])[::-1]
        room_y = np.append(room.corners[1][3:], room.corners[1][0])[::-1]
        min_x = min(room_x)
        max_x = max(room_x)
        self.positions = []

        possible_xs = np.arange(min_x + 0.1, max_x - 0.1, 0.3)
        xs = np.random.choice(possible_xs, n_sound, replace=False)
        for x in xs:
            offset = np.random.uniform(0.1, 0.6)
            idx = np.searchsorted(room_x, x, side="right")
            if room_x[idx] == x:
                self.positions.append([x, room_y[idx] + offset])
            else:
                x1 = room_x[idx - 1]
                y1 = room_y[idx - 1]
                x2 = room_x[idx]
                y2 = room_y[idx]
                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                self.positions.append([x, y + offset])


class Ambient(AudioLoader):
    def __init__(self, config, n_sound, fs):
        super().__init__(config, n_sound, fs)
        self.snr = config["snr"]


class Drone(AudioLoader):
    def __init__(self, config, n_sound, fs):
        super().__init__(config, n_sound, fs)
        self.snr = config["snr"]
        config_mic_positions = config["mic_positions"]
        self.mic_positions = self._create_mic_positions(config_mic_positions)
        config_propeller = config.get("propeller", {})
        self.offset = np.array(config_propeller.get("offset", [0, 0]))
        self.width = config_propeller.get("width", 0.1)
        # configにsourceのpositionを追加
        self._adjust_source_positions(config, (0, config_mic_positions["height"]))
        # ドローンは1つの音源として扱う
        self.n_sound = 1

    def _create_mic_positions(self, config):
        return pra.circular_2D_array(
            center=(0, config["height"]),
            M=config["M"],
            phi0=0,
            radius=config["radius"],
        )

    def _adjust_source_positions(self, config, center):
        num_sources = self.n_sound
        if num_sources == 0:
            return
        xs = np.linspace(-self.width / 2, self.width / 2, num_sources)
        self.positions = []
        for x in xs:
            self.positions.append((x + center[0] + self.offset[0], center[1] + self.offset[1]))
