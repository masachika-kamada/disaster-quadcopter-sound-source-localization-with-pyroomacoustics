from glob import glob

import numpy as np
import pyroomacoustics as pra

from .file_io import load_signal_from_wav


class AudioLoader:
    _x_positions_pool = None

    @classmethod
    def initialize_x_positions_pool(cls, room, step=0.5):
        room_x = np.append(room.corners[0][3:], room.corners[0][0])[::-1]
        min_x = min(room_x)
        max_x = max(room_x)
        cls._x_positions_pool = list(np.arange(min_x + 0.25, max_x - 0.24, step))

    @classmethod
    def get_x_positions(cls, n_sound, seed):
        if cls._x_positions_pool is None or len(cls._x_positions_pool) < n_sound:
            raise ValueError("Not enough x positions available")
        np.random.seed(seed)
        xs = np.random.choice(cls._x_positions_pool, n_sound, replace=False)
        cls._x_positions_pool = [x for x in cls._x_positions_pool if x not in xs]
        return xs

    def __init__(self, config, n_sound, seed, fs=16000):
        self.n_sound = n_sound
        self.signals = []
        source_dir = config["source_dir"]
        np.random.seed(seed)
        source_files = np.random.choice(glob(f"{source_dir}/*.wav"), n_sound, replace=False)
        for file_path in source_files:
            signal = load_signal_from_wav(file_path, fs)
            self.signals.append(signal)


class PositionedAudioLoader(AudioLoader):
    @staticmethod
    def calculate_positions(room, n_sound, seed):
        xs = AudioLoader.get_x_positions(n_sound, seed)
        room_x = np.append(room.corners[0][3:], room.corners[0][0])[::-1]
        room_y = np.append(room.corners[1][3:], room.corners[1][0])[::-1]

        positions = []
        for x in xs:
            np.random.seed(seed)
            offset = np.random.uniform(0.1, 0.5)
            idx = np.searchsorted(room_x, x, side="right")
            if room_x[idx] == x:
                positions.append([x, room_y[idx] + offset])
            else:
                x1 = room_x[idx - 1]
                y1 = room_y[idx - 1]
                x2 = room_x[idx]
                y2 = room_y[idx]
                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                positions.append([x, y + offset])
        return positions


class Voice(PositionedAudioLoader):
    def __init__(self, config, n_sound, seed, fs, room):
        super().__init__(config, n_sound, seed, fs)
        self.positions = self.calculate_positions(room, n_sound, seed)


class Ambient(PositionedAudioLoader):
    def __init__(self, config, n_sound, seed, fs, room):
        super().__init__(config, n_sound, seed, fs)
        self.snr = config["snr"]
        self.positions = self.calculate_positions(room, n_sound, seed)


class Drone(AudioLoader):
    def __init__(self, config, seed, fs):
        super().__init__(config, 4, seed, fs)
        config_mic_positions = config["mic_positions"]
        self.mic_positions = self._create_mic_positions(config_mic_positions)
        config_propeller = config.get("propeller", {})
        self.offset = np.array(config_propeller.get("offset", [0, 0]))
        self.width = config_propeller.get("width", 0.1)
        self._adjust_source_positions((0, config_mic_positions["height"]))

        d_ground = config_mic_positions["height"]
        d_propeller = self.offset[1]
        snr_diff = 10 * np.log10(d_propeller ** 2 / d_ground ** 2)
        self.snr = config["snr"] + snr_diff

    def _create_mic_positions(self, config):
        center = (0, config["height"])
        radius = config["radius"]
        mic_positions = pra.circular_2D_array(center=center, M=8, phi0=0, radius=radius)
        # 上側のマイクロホンを削除
        mic_positions = np.delete(mic_positions, np.where(mic_positions[1] > center[1]), axis=1)
        # 新しい座標を既存の配列に追加
        new_points = np.array([
            [center[0] - radius / 2, center[1]],
            center,
            [center[0] + radius / 2, center[1]],
            [center[0], center[1] - radius / 2],
        ]).T
        mic_positions = np.concatenate((mic_positions, new_points), axis=1)
        return mic_positions

    def _adjust_source_positions(self, center):
        num_sources = self.n_sound
        if num_sources == 0:
            return
        xs = np.linspace(-self.width / 2, self.width / 2, num_sources)
        self.positions = []
        for x in xs:
            self.positions.append((x + center[0] + self.offset[0],
                                   center[1] + self.offset[1]))
        self.n_sound = 1
