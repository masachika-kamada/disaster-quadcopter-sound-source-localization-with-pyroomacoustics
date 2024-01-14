import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra


class Room:
    def __init__(self, config):
        np.random.seed(0)
        config_room = config["room"]
        room_dim = config_room["room_dim"]
        corners = np.array([
            [int(-room_dim[0] / 2), 0],
            [int(-room_dim[0] / 2), room_dim[1]],
            [int( room_dim[0] / 2), room_dim[1]],
            [int( room_dim[0] / 2), 0],
        ]).T
        self.corners = self._generate_floor(corners, config_room["floor"])
        self.fs = config_room["fs"]
        self.max_order = config_room["max_order"]

        self.rooms = {}
        materials = self._load_materials(len(self.corners[0]), config_room["floor"]["material"])
        m_none = self._load_materials(len(self.corners[0]), None)
        for name, max_order, materials in zip(
            ["source", "ncm_rev", "ncm_dir"],
            [self.max_order, self.max_order, 0],
            [materials, materials, m_none]
        ):
            self.rooms[name] = self._create_room(self.corners, max_order, materials)

    def _generate_floor(self, corners, config):
        shape = config["shape"]
        if shape == "flat":
            return corners

        corners = corners.T  # [x, y]
        x_min = corners[0, 0]
        x_max = corners[-1, 0]
        y = corners[0, 1]

        if shape in ["triangle", "square"]:
            new_corners = []
            interval = config["interval"]
            height = config["height"]
            n_rough = int((x_max - x_min) / interval)
            interval = (x_max - x_min) / n_rough

            for i in range(1, n_rough):
                x = x_max - i * interval
                if shape == "triangle":
                    if i % 2 == 0:
                        new_corners.append([x, y])
                    else:
                        new_corners.append([x, y + height])
                elif shape == "square":
                    if i % 2 == 0:
                        new_corners.append([x, y + height])
                        new_corners.append([x, y])
                    else:
                        new_corners.append([x, y])
                        new_corners.append([x, y + height])
            new_corners = np.array(new_corners)

        elif shape == "random":
            min_interval, max_interval = config["roughness"]
            n_max = int((x_max - x_min) // min_interval - 1)
            x_rand = np.random.rand(n_max) * (max_interval - min_interval) + min_interval
            x_rand = x_max - np.cumsum(x_rand)
            x_rand = x_rand[x_rand >= x_min + min_interval]
            y_rand = np.random.rand(len(x_rand)) * max_interval * 0.5 - max_interval * 0.25
            new_corners = np.vstack([x_rand, y_rand]).T

        new_corners = np.vstack([corners, new_corners])
        return new_corners.T

    def _load_materials(self, num_corners: int, floor_material):
        floor_material = self._create_materials(floor_material)
        no_wall_material = self._create_materials()
        materials = [floor_material] + [no_wall_material] * 3 + [floor_material] * (num_corners - 4)
        # listを逆順にする
        materials = materials[::-1]
        return materials

    def _create_materials(self, m=None):
        return pra.Material(energy_absorption=1.0) if m is None else pra.Material(m)

    def _create_room(self, corners, max_order, materials):
        return pra.Room.from_corners(corners, fs=self.fs, max_order=max_order, materials=materials)

    def place_microphones(self, mic_positions):
        for room in self.rooms.values():
            room.add_microphone_array(pra.MicrophoneArray(mic_positions, self.fs))

    def place_source(self, voice, drone, ambient=None):
        for signal, position in zip(voice.signals, voice.positions):
            self.rooms["source"].add_source(position, signal=signal)
        for signal, position in zip(drone.signals, drone.positions):
            for room_name in ["source", "ncm_rev", "ncm_dir"]:
                self.rooms[room_name].add_source(position, signal=signal)
        if ambient is not None:
            for signal, position in zip(ambient.signals, ambient.positions):
                self.rooms["source"].add_source(position, signal=signal)

    def simulate(self, output_dir):
        for room_name in ["source", "ncm_rev", "ncm_dir"]:
            self.rooms[room_name].simulate()

        self.rooms["source"].plot()
        plt.savefig(f"{output_dir}/room.png")
        plt.close()

        return (self.rooms["source"].mic_array.signals,
                self.rooms["ncm_rev"].mic_array.signals,
                self.rooms["ncm_dir"].mic_array.signals)
