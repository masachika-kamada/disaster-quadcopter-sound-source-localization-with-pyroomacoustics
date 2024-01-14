import numpy as np
import csv
import os
from lib.doa import detect_peaks


def export_metrics(output_dir: str, music_spectra: list, ans: dict):
    music_spectra = np.array(music_spectra)
    ans_voice = np.array(ans["voice"])
    ans_ambient = np.array(ans["ambient"]) if "ambient" in ans else []

    d_theta = np.pi / 180

    # Peak
    music_spectra = music_spectra[:, :180]
    music_spectra /= np.max(music_spectra)
    # Initialize variables for metrics
    TP_voice, FN_voice, TP_ambient, FN_ambient, FP = 0, 0, 0, 0, 0

    for spectrum in music_spectra:
        median = np.median(spectrum)
        idx = detect_peaks(spectrum, mph=median, show=False)

        _TP_voice, _FN_voice, _TP_ambient, _FN_ambient, _FP = \
            calculate_evaluation_metrics(ans_voice, ans_ambient, - np.pi + idx * d_theta, 3)
        TP_voice += _TP_voice
        FN_voice += _FN_voice
        if len(ans_ambient) > 0:
            TP_ambient += _TP_ambient
            FN_ambient += _FN_ambient
        FP += _FP

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
                             "TP_voice", "FN_voice", "TP_ambient", "FN_ambient", "FP"])

        # メトリクスとパラメータを書き込む
        writer.writerow(params + [method, TP_voice, FN_voice, TP_ambient, FN_ambient, FP])


def calculate_evaluation_metrics(true_values_voice, true_values_ambient, predicted_values, tolerance_deg=3):
    tolerance_rad = np.radians(tolerance_deg)

    tv_voice_used = [False] * len(true_values_voice)
    tv_ambient_used = [False] * len(true_values_ambient)
    pv_used = [False] * len(predicted_values)

    for i, tv in enumerate(true_values_voice):
        for j, pv in enumerate(predicted_values):
            if tv - tolerance_rad <= pv <= tv + tolerance_rad:
                tv_voice_used[i] = True
                pv_used[j] = True

    for i, tv in enumerate(true_values_ambient):
        for j, pv in enumerate(predicted_values):
            if tv - tolerance_rad <= pv <= tv + tolerance_rad:
                tv_ambient_used[i] = True
                pv_used[j] = True

    TP_voice = sum(tv_voice_used)
    FN_voice = len(tv_voice_used) - TP_voice
    TP_ambient = sum(tv_ambient_used) if len(tv_ambient_used) > 0 else None
    FN_ambient = len(tv_ambient_used) - TP_ambient if len(tv_ambient_used) > 0 else None
    FP = sum([not pv for pv in pv_used])

    # # それぞれどの角度で発生したか
    # rad_TP_voice = [tv for tv, used in zip(true_values_voice, tv_voice_used) if used]
    # rad_TP_ambient = [tv for tv, used in zip(true_values_ambient, tv_ambient_used) if used] \
    #     if len(tv_ambient_used) > 0 else None
    # rad_FN_voice = [tv for tv, used in zip(true_values_voice, tv_voice_used) if not used]
    # rad_FN_ambient = [tv for tv, used in zip(true_values_ambient, tv_ambient_used) if not used] \
    #     if len(tv_ambient_used) > 0 else None
    # rad_FP = [pv for pv, used in zip(predicted_values, pv_used) if not used]

    return TP_voice, FN_voice, TP_ambient, FN_ambient, FP  #, \
           # rad_TP_voice, rad_TP_ambient, rad_FN_voice, rad_FN_ambient, rad_FP
