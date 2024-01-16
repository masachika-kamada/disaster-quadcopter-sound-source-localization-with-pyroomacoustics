import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import pyroomacoustics as pra


def plot_decomposed_values(decomposed_values, display, save):
    fig, axes = plt.subplots(1, 8, figsize=(15, 8), sharey=True)
    user_input = [None]  # ユーザー入力を格納するためのリスト

    for i in range(8):
        axes[i].plot(
            decomposed_values[..., i], label=str(i + 1), marker="o", linestyle=""
        )
        axes[i].set_title(f"Value {i+1}")
        axes[i].set_xlabel("Row Index")
        axes[i].set_ylim([np.min(decomposed_values), np.max(decomposed_values)])

    axes[0].set_ylabel("Value Magnitude")
    plt.suptitle("Distribution of Decomposed Values Magnitudes Across Rows")

    # テキストボックスの追加
    axbox = plt.axes([0.93, 0.06, 0.03, 0.04])
    text_box = TextBox(axbox, "N:")

    # ユーザー入力を取得するコールバック関数
    def submit(text):
        user_input[0] = text
        plt.close(fig)

    text_box.on_submit(submit)

    if save:
        plt.savefig(f"{save}.png")
    if display:
        plt.show()
    plt.close()

    return user_input[0]  # ユーザー入力値を返す


def plot_music_spectra(spectra, output_dir: str) -> None:
    # 2行4列のサブプロットを作成
    fig = plt.figure(figsize=(18, 9))

    # 上側の行に極座標系グラフを配置
    rad = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    for i in range(4):
        ax = fig.add_subplot(2, 4, i + 1, polar=True)
        ax.plot(rad, spectra[i], color="blue", alpha=0.2)
        ax.set_title(f"MUSIC Spectrum {i + 1} (Polar)")
        ax.grid(True)

    # 下側の行に直交座標系グラフを配置
    x = np.linspace(-180, 0, 180)
    for i in range(4):
        ax = fig.add_subplot(2, 4, i + 4 + 1)
        # ax.plot(x, spectra[i][:180][:, 14], color="blue", alpha=0.2)
        ax.plot(x, spectra[i][:180], color="blue", alpha=0.2)
        # 平均値
        avg = np.mean(spectra[i][:180], axis=1) * 10
        ax.plot(x, avg, color="red")
        ax.set_title(f"MUSIC Spectrum {i + 1} (Cartesian)")
        ax.set_xticks(np.arange(-180, 1, 30))
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def compute_spatial_spectrum(noise_subspace):
    num_freq = 102
    spatial_spectrum = np.zeros((num_freq, 360))
    steering_vector = compute_steering_vector()
    for f in range(num_freq):
        steering_vector_f = steering_vector[f, :, :]
        noise_subspace_f = noise_subspace[f, :, :]
        # calculate the MUSIC spectrum for each angle and frequency bin
        vector_norm_squared = np.abs(np.sum(np.conj(steering_vector_f) * steering_vector_f, axis=1))
        noise_projection = np.abs(np.sum(
            np.conj(steering_vector_f) @ noise_subspace_f * (noise_subspace_f.conj().T @ steering_vector_f.T).T,
            axis=1))
        spatial_spectrum[f, :] = vector_norm_squared / noise_projection
    return spatial_spectrum.T


def compute_steering_vector(angle_step=1):
    L = get_mic_positions()
    n_channels = L.shape[1]
    theta = np.deg2rad(np.arange(-180, 180, angle_step))

    direction_vectors = np.zeros((2, len(theta)), dtype=float)
    direction_vectors[0, :] = np.cos(theta)
    direction_vectors[1, :] = np.sin(theta)

    start_freq = 312.5
    end_freq = 3468.75
    step = 31.25
    freq_hz = np.arange(start_freq, end_freq + step, step)

    steering_phase = np.einsum(
        "k,ism,ism->ksm",
        2.0j * np.pi / 343.0 * freq_hz,
        direction_vectors[..., None],
        L[:, None, :]
    )
    steering_vector = 1.0 / np.sqrt(n_channels) * np.exp(steering_phase)
    return steering_vector


def get_mic_positions():
    center = (0, 0)
    radius = 0.2

    mic_positions = pra.circular_2D_array(center=center, M=8, phi0=0, radius=radius)
    # 上側のマイクロホンを削除
    mic_positions = np.delete(mic_positions, np.where(mic_positions[1] > center[1] + 0.01), axis=1)
    # 新しい座標を既存の配列に追加
    new_points = np.array([
        [center[0] - radius / 2, center[1]],
        center,
        [center[0] + radius / 2, center[1]],
        [center[0], center[1] - radius / 2],
    ]).T
    mic_positions = np.concatenate((mic_positions, new_points), axis=1)
    return mic_positions


def main():
    dir_ref = "experiments/5;[0.1, 1.0];hard_surface;3;0;11;0/GEVD_incremental"
    decomposed_values = np.load(f"{dir_ref}/decomposed_values.npy")
    decomposed_vectors = np.load(f"{dir_ref}/decomposed_vectors.npy")

    print(f"{decomposed_values.shape=}")
    print(f"{decomposed_vectors.shape=}")

    for value, vector in zip(decomposed_values, decomposed_vectors):
        spectra = []
        # n = plot_decomposed_values(value, True, None)
        for i in range(1, 5):  # nの値を1から4に変更
            noise_subspace = vector[..., :-i]
            spatial_spectrum = compute_spatial_spectrum(noise_subspace)
            spectra.append(spatial_spectrum)

        plot_music_spectra(spectra, None)
        # break


if __name__ == "__main__":
    main()
