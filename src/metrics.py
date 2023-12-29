import numpy as np
import csv
import os
from lib.doa import detect_peaks


def export_metrics(output_dir: str, music_spectra: list, ans: dict):
    music_spectra = np.array(music_spectra)
    ans_voice = np.array(ans["voice"])
    ans_ambient = np.array(ans["ambient"]) if "ambient" in ans else []

    # Area Inside a Polar Curve
    music_spectra /= np.max(music_spectra)
    d_theta = np.pi / 180
    aipc_lower, aipc_upper = 0, 0
    for spectrum in music_spectra:
        aipc_upper += calculate_aipc(spectrum[:180], d_theta)
        aipc_lower += calculate_aipc(spectrum[180:], d_theta)
    aipc_lower = round(aipc_lower/ len(music_spectra), 5)
    aipc_upper = round(aipc_upper/ len(music_spectra), 5)

    # Peak
    music_spectra = music_spectra[:, :180]
    music_spectra /= np.max(music_spectra)
    # Initialize variables for metrics
    TP_voice_3, FN_voice_3, TP_ambient_3, FN_ambient_3, FP_3 = 0, 0, 0, 0, 0
    TP_voice_5, FN_voice_5, TP_ambient_5, FN_ambient_5, FP_5 = 0, 0, 0, 0, 0

    for spectrum in music_spectra:
        median = np.median(spectrum)
        idx = detect_peaks(spectrum, mph=median, show=False)

        TP_voice, FN_voice, TP_ambient, FN_ambient, FP = \
            calculate_evaluation_metrics(ans_voice, ans_ambient, - np.pi + idx * d_theta, 3)
        TP_voice_3 += TP_voice
        FN_voice_3 += FN_voice
        TP_ambient_3 += TP_ambient
        FN_ambient_3 += FN_ambient
        FP_3 += FP

        TP_voice, FN_voice, TP_ambient, FN_ambient, FP = \
            calculate_evaluation_metrics(ans_voice, ans_ambient, - np.pi + idx * d_theta, 5)
        TP_voice_5 += TP_voice
        FN_voice_5 += FN_voice
        TP_ambient_5 += TP_ambient
        FN_ambient_5 += FN_ambient
        FP_5 += FP

    # export
    params = output_dir.split("/")[-2].split(";")
    method = output_dir.split("/")[-1]

    csv_file_path = "results/metrics.csv"
    is_file_exist = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # ファイルが存在しない場合、または空の場合、ヘッダーを書き込む
        if not is_file_exist or os.path.getsize(csv_file_path) == 0:
            writer.writerow(["height", "roughness", "material", "n_voice", "n_ambient", \
                             "snr_ego", "snr_ambient", "method", \
                             "aipc_lower", "aipc_upper", \
                             "TP_voice_3", "FN_voice_3", "TP_ambient_3", "FN_ambient_3", "FP_3", \
                             "TP_voice_5", "FN_voice_5", "TP_ambient_5", "FN_ambient_5", "FP_5"])

        # メトリクスとパラメータを書き込む
        writer.writerow(params + [method, aipc_lower, aipc_upper, \
                                  TP_voice_3, FN_voice_3, TP_ambient_3, FN_ambient_3, FP_3, \
                                  TP_voice_5, FN_voice_5, TP_ambient_5, FN_ambient_5, FP_5])


def calculate_aipc(vals: np.ndarray, d_theta: float):
    """Area Inside a Polar Curve"""
    return np.sum(vals[:-1] * vals[1:]) * d_theta / 2


def calculate_evaluation_metrics(true_values_voice, true_values_ambient, predicted_values, tolerance_deg=3):
    tolerance_rad = np.radians(tolerance_deg)

    tv_voice_rec = [False] * len(true_values_voice)
    tv_ambient_rec = [False] * len(true_values_ambient)
    pv_rec = [False] * len(predicted_values)

    for i, tv in enumerate(true_values_voice):
        for j, pv in enumerate(predicted_values):
            if tv - tolerance_rad <= pv <= tv + tolerance_rad:
                tv_voice_rec[i] = True
                pv_rec[j] = True

    for i, tv in enumerate(true_values_ambient):
        for j, pv in enumerate(predicted_values):
            if tv - tolerance_rad <= pv <= tv + tolerance_rad:
                tv_ambient_rec[i] = True
                pv_rec[j] = True

    TP_voice = sum(tv_voice_rec)
    FN_voice = len(tv_voice_rec) - TP_voice
    TP_ambient = sum(tv_ambient_rec) if len(tv_ambient_rec) > 0 else None
    FN_ambient = len(tv_ambient_rec) - TP_ambient if len(tv_ambient_rec) > 0 else None
    FP = sum([not pv for pv in pv_rec])

    return TP_voice, FN_voice, TP_ambient, FN_ambient, FP
