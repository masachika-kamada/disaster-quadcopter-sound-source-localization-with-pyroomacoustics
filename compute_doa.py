import argparse
import os

import numpy as np
from tqdm import tqdm

from lib.custom import create_doa_object, perform_fft_on_frames
from src.class_sound import Drone
from src.file_io import load_config, load_signal_from_npz
from src.visualization_tools import plot_music_spectra


def main(args, config, experiment_dir):
    config_drone = config["drone"]
    fs = config["pra"]["room"]["fs"]
    drone = Drone(config_drone, fs=fs)

    signal_source, fs = load_signal_from_npz(f"{experiment_dir}/simulation/source.npz")
    signal_ncm_rev, fs = load_signal_from_npz(f"{experiment_dir}/simulation/ncm_rev.npz")
    signal_ncm_dir, fs = load_signal_from_npz(f"{experiment_dir}/simulation/ncm_dir.npz")

    X_source = perform_fft_on_frames(signal_source, args.window_size, args.hop_size)
    X_ncm_rev = perform_fft_on_frames(signal_ncm_rev, args.window_size, args.hop_size)
    X_ncm_dir = perform_fft_on_frames(signal_ncm_dir, args.window_size, args.hop_size)

    num_voice = int(experiment_dir.split("-")[3])
    num_ambient = int(experiment_dir.split("-")[4])
    num_src = num_voice + num_ambient + 1
    frame_length = 100

    # SEVD
    method = "SEVD"
    output_dir = f"{experiment_dir}/{method}"
    os.makedirs(output_dir, exist_ok=True)
    doa = create_doa_object(
        method=method,
        source_noise_thresh=100,
        mic_positions=drone.mic_positions,
        fs=fs,
        nfft=args.window_size,
        num_src=num_src,
        output_dir=output_dir,
    )
    for f in range(0, X_source.shape[2], frame_length // 4):
        xs = X_source[:, :, f : f + frame_length]
        doa.locate_sources(xs, None, freq_range=args.freq_range, auto_identify=True)
    plot_music_spectra(doa, output_dir=output_dir)
    np.save(f"{output_dir}/decomposed_values.npy", np.array(doa.decomposed_values_strage))
    np.save(f"{output_dir}/decomposed_vectors.npy", np.array(doa.decomposed_vectors_strage))

    # GEVD
    method = "GEVD"
    doa = create_doa_object(
        method=method,
        source_noise_thresh=100,
        mic_positions=drone.mic_positions,
        fs=fs,
        nfft=args.window_size,
        num_src=num_src,
        output_dir="",
    )

    # incremental
    frame_s = 140
    frame_t_n = 90
    output_dir = f"{experiment_dir}/{method}_incremental"
    doa.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    for f in range(0, X_source.shape[2] - frame_s - frame_t_n, frame_length // 4):
        xn = X_source[:, :, f : f + frame_t_n]
        f2 = f + frame_s + frame_t_n
        xs = X_source[:, :, f2 : f2 + frame_length]
        doa.locate_sources(xs, xn, freq_range=args.freq_range, auto_identify=True)
    plot_music_spectra(doa, output_dir=output_dir)
    np.save(f"{output_dir}/decomposed_values.npy", np.array(doa.decomposed_values_strage))
    np.save(f"{output_dir}/decomposed_vectors.npy", np.array(doa.decomposed_vectors_strage))

    # ans
    for basename, X_ncm in zip(["rev", "dir"], [X_ncm_rev, X_ncm_dir]):
        output_dir = f"{experiment_dir}/{method}_ans_{basename}"
        doa.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        for f in range(0, X_source.shape[2], frame_length // 4):
            xs = X_source[:, :, f : f + frame_length]
            xn = X_ncm[:, :, f : f + frame_length]
            doa.locate_sources(xs, xn, freq_range=args.freq_range, auto_identify=True)
        plot_music_spectra(doa, output_dir=output_dir)
        np.save(f"{output_dir}/decomposed_values.npy", np.array(doa.decomposed_values_strage))
        np.save(f"{output_dir}/decomposed_vectors.npy", np.array(doa.decomposed_vectors_strage))

    # diff
    for basename, X_ncm in zip(["rev", "dir"], [X_ncm_rev, X_ncm_dir]):
        output_dir = f"{experiment_dir}/{method}_diff_{basename}"
        doa.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        for f in range(0, X_source.shape[2], frame_length // 4):
            xs = X_source[:, :, f : f + frame_length]
            xn = X_ncm[:, :, f : f + frame_length]
            doa.locate_sources(xs, xn, freq_range=args.freq_range, auto_identify=True, ncm_diff=0.05)
        plot_music_spectra(doa, output_dir=output_dir)
        np.save(f"{output_dir}/decomposed_values.npy", np.array(doa.decomposed_values_strage))
        np.save(f"{output_dir}/decomposed_vectors.npy", np.array(doa.decomposed_vectors_strage))

    # stable
    for basename, X_ncm in zip(["rev", "dir"], [X_ncm_rev, X_ncm_dir]):
        output_dir = f"{experiment_dir}/{method}_stable_{basename}"
        doa.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        for f in range(0, X_source.shape[2], frame_length // 4):
            xs = X_source[:, :, f : f + frame_length]
            doa.locate_sources(xs, X_ncm, freq_range=args.freq_range, auto_identify=True)
        plot_music_spectra(doa, output_dir=output_dir)
        np.save(f"{output_dir}/decomposed_values.npy", np.array(doa.decomposed_values_strage))
        np.save(f"{output_dir}/decomposed_vectors.npy", np.array(doa.decomposed_vectors_strage))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate room acoustics based on YAML config.")
    parser.add_argument("--window_size", default=512, type=int, help="Window size for FFT")
    parser.add_argument("--hop_size", default=128, type=int, help="Hop size for FFT")
    parser.add_argument("--freq_range", default=[300, 3500], type=int, nargs=2, help="Frequency range for DoA")
    parser.add_argument("--source_noise_thresh", default=100, type=int, help="Frequency range for DoA")
    args = parser.parse_args()

    config = load_config("experiments/config.yaml")

    base_path = "experiments"
    experiment_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    for experiment_dir in tqdm(experiment_dirs, desc="Processing experiments"):
        main(args, config, os.path.join(base_path, experiment_dir))
