import numpy as np
import csv
import os
from lib.doa import detect_peaks


def export_metrics(output_dir: str, music_spectra: list, ans: list):
    music_spectra = np.array(music_spectra)
    ans = np.array(ans)

    # Area Inside a Polar Curve
    music_spectra /= np.max(music_spectra)
    d_theta = np.pi / 180
    aipc_pos, aipc_neg = 0, 0
    for spectrum in music_spectra:
        aipc_neg += calculate_aipc(spectrum[:180], d_theta)
        aipc_pos += calculate_aipc(spectrum[180:], d_theta)
    aipc_pos /= len(music_spectra)
    aipc_neg /= len(music_spectra)

    # Peak
    music_spectra = music_spectra[:, :180]
    music_spectra /= np.max(music_spectra)
    total_TP, total_FP, total_FN = 0, 0, 0

    for spectrum in music_spectra:
        median = np.median(spectrum)
        idx = detect_peaks(spectrum, mph=median, show=False)

        TP, FP, FN = calculate_evaluation_metrics(ans, - np.pi + idx * d_theta)
        total_TP += TP
        total_FP += FP
        total_FN += FN

    recall = total_TP / (total_TP + total_FN) if total_TP + total_FN > 0 else 0
    precision = total_TP / (total_TP + total_FP) if total_TP + total_FP > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    for metric in [recall, precision, f1, aipc_pos, aipc_neg]:
        metric = round(metric, 3)

    params = output_dir.split("/")[-2].split(";")
    method = output_dir.split("/")[-1]

    # CSVファイルに書き込み
    csv_file_path = "metrics.csv"
    is_file_exist = os.path.isfile(csv_file_path)

    with open(csv_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        # ファイルが存在しない場合、または空の場合、ヘッダーを書き込む
        if not is_file_exist or os.path.getsize(csv_file_path) == 0:
            writer.writerow(["heights", "roughnesses", "materials", "n_voices", "n_ambients", \
                             "snr_egos", "snr_ambients", "idx", "method", \
                             "recall", "precision", "f1", "aipc_pos", "aipc_neg"])

        # メトリクスとパラメータを書き込む
        writer.writerow(params + [method, round(recall, 3), round(precision, 3), round(f1, 3), \
                                  round(aipc_pos, 5), round(aipc_neg, 5)])


def calculate_aipc(vals: np.ndarray, d_theta: float):
    """Area Inside a Polar Curve"""
    return np.sum(vals[:-1] * vals[1:]) * d_theta / 2


def calculate_evaluation_metrics(true_values, predicted_values, tolerance_deg=3):
    tolerance_rad = np.radians(tolerance_deg)

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = len(true_values)  # Initially, all true values are considered as False Negatives

    used_predictions = set()

    for true_angle in true_values:
        closest_prediction = None
        min_distance = float("inf")

        for i, predicted_angle in enumerate(predicted_values):
            distance = np.abs(true_angle - predicted_angle)
            if distance <= tolerance_rad and distance < min_distance and i not in used_predictions:
                closest_prediction = i
                min_distance = distance

        if closest_prediction is not None:
            TP += 1
            FN -= 1
            used_predictions.add(closest_prediction)

    FP = len(predicted_values) - len(used_predictions)

    return TP, FP, FN
