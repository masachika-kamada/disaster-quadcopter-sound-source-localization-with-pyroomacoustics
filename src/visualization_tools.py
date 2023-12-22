from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
from IPython.display import Audio, display


def plot_room(room: pra.ShoeBox) -> None:
    room_dim = room.get_bbox()[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    room.plot(ax=ax)

    # プロット範囲を部屋の大きさに合わせる
    ax.set_xlim([0, room_dim[0]])
    ax.set_ylim([0, room_dim[1]])
    ax.set_zlim([0, room_dim[2]])
    ax.set_box_aspect(room_dim)
    plt.show()


def plot_room_views(room: pra.ShoeBox,
                    zoom_center: Optional[List[float]] = None,
                    zoom_size: Optional[float] = None) -> None:
    # Get the room dimensions from the bounding box
    room_dim = room.get_bbox()[:, 1]

    fig = plt.figure(figsize=(15, 6))
    views = [(90, -90, "Top View"), (0, -90, "Front View"), (0, 0, "Side View")]

    for i, (elev, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        room.plot(fig=fig, ax=ax)
        ax.view_init(elev, azim)
        ax.set_title(title)
        ax.set_xlim([0, room_dim[0]])
        ax.set_ylim([0, room_dim[1]])
        ax.set_zlim([0, room_dim[2]])

        if zoom_center is not None and zoom_size is not None:
            ax.set_xlim([zoom_center[0] - zoom_size / 2, zoom_center[0] + zoom_size / 2])
            ax.set_ylim([zoom_center[1] - zoom_size / 2, zoom_center[1] + zoom_size / 2])
            ax.set_zlim([zoom_center[2] - zoom_size / 2, zoom_center[2] + zoom_size / 2])
        else:
            ax.set_xlim([0, room_dim[0]])
            ax.set_ylim([0, room_dim[1]])
            ax.set_zlim([0, room_dim[2]])

        ax.set_box_aspect(room_dim)
    plt.show()


def play_audio(audio_data: np.ndarray, fs: int) -> None:
    display(Audio(audio_data, rate=fs))


def plot_music_spectra(doa, output_dir: str) -> None:
    estimated_angles = doa.grid.azimuth
    music_spectra = doa.spectra_storage

    for i in range(len(music_spectra)):
        plt.polar(estimated_angles, music_spectra[i], color="blue", alpha=0.2)

    plt.title("MUSIC Spectrum (Polar Coordinates)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/music_spectrum.png")
    plt.close()


def plot_reverberation_wall(room: pra.Room, filename: str) -> None:
    fig, ax = plt.subplots()
    walls = room.walls
    x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')

    for wall in walls:
        xs, ys = wall.corners
        absorption = wall.absorption

        x_min, x_max = min(x_min, xs.min()), max(x_max, xs.max())
        y_min, y_max = min(y_min, ys.min()), max(y_max, ys.max())

        if np.array(absorption).mean() != 1:
            color = "red"
            ax.plot(xs, ys, color=color)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.savefig(filename)
    plt.close(fig)
