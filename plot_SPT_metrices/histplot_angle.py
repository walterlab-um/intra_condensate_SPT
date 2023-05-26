import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

color_palette = [
    "#001219",
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#e9d8a6",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
    "#ae2012",
    "#9b2226",
]
os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
N_bins = 6
# threshold 1: minimla tracklength, because too short a track is too stochastic in angle distribution
threshold_tracklength = 50
# threshold 3: D error bounds to determine static molecule
s_per_frame = 0.02
static_err = 0.016
threshold_log10D = np.log10(static_err**2 / (4 * (s_per_frame)))
# threshold 4: Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um

dict_input_path = {
    "0Dex, -, 0h": "SPT_results_AIO-0Dex_noTotR_0h.csv",
    "0Dex, -, 3h": "SPT_results_AIO-0Dex_noTotR_3h.csv",
    "0Dex, He, 1h": "SPT_results_AIO-0Dex_Hela_1h.csv",
    "0Dex, Ce, 1h": "SPT_results_AIO-0Dex_Cerebral_1h.csv",
    "0Dex, Sp, 1h": "SPT_results_AIO-0Dex_Spinal_1h.csv",
    "10Dex, -, 0h": "SPT_results_AIO-10Dex_noTotR_0h.csv",
    "10Dex, -, 3h": "SPT_results_AIO-10Dex_noTotR_3h.csv",
    "10Dex, He, 0h": "SPT_results_AIO-10Dex_Hela_0h.csv",
    "10Dex, Ce, 0h": "SPT_results_AIO-10Dex_Cerebral_0h.csv",
    "10Dex, Sp, 0h": "SPT_results_AIO-10Dex_Spinal_0h.csv",
}
lst_keys = list(dict_input_path.keys())
for i in track(range(len(lst_keys))):
    key = lst_keys[i]
    color = color_palette[i]
    df_current_file = pd.read_csv(dict_input_path[key])
    df_current_file = df_current_file.astype(
        {
            "linear_fit_log10D": float,
            "N_steps": int,
            "Displacement_um": float,
        }
    )
    df_longtracks = df_current_file[df_current_file["N_steps"] >= threshold_tracklength]
    df_mobile_byD = df_longtracks[
        df_longtracks["linear_fit_log10D"] >= threshold_log10D
    ]
    df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]

    # Per Track
    # first five columns are filename, trackID, N_steps, total disp, list of angles. Thus, # of bins = # columes - 5 + 1
    bins = np.linspace(0, 180, N_bins + 1).astype(int)

    hist_per_track = df_mobile.iloc[:, -N_bins:].to_numpy(dtype=float)
    hist_per_track_mean = np.nanmean(hist_per_track, axis=0)
    hist_per_track_std = np.nanstd(hist_per_track, axis=0)

    plt.figure(figsize=(5, 4), dpi=300)
    plt.errorbar(
        x=bins[:-1] + (bins[1] - bins[0]) / 2,
        y=hist_per_track_mean,
        yerr=hist_per_track_std,
        ls="-",
        color=color,
        lw=2,
        capsize=5,
        capthick=3,
    )
    plt.title(key + " Per Track", weight="bold")
    plt.xlabel("Angle between Two Steps, Degree", weight="bold")
    plt.ylabel("Probability", weight="bold")
    plt.ylim(0, 0.5)
    plt.xlim(0, 180)
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig(
        "AngleHist_PerTrack_" + dict_input_path[key][16:-4] + ".png", format="png"
    )
    plt.close()

    # Per step
    lst_angle_arrays = []
    for array_like_string in df_mobile["list_of_angles"].to_list():
        lst_angle_arrays.append(
            np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        )
    all_angles = np.concatenate(lst_angle_arrays)
    plt.figure(figsize=(5, 4), dpi=300)
    plt.hist(
        all_angles,
        bins=bins,
        weights=np.ones_like(all_angles) / len(all_angles),
        color=color,
    )
    plt.title(key + " Per Step", weight="bold")
    plt.xlabel("Angle between Two Steps, Degree", weight="bold")
    plt.ylabel("Probability", weight="bold")
    plt.ylim(0, 0.3)
    plt.xlim(0, 180)
    plt.xticks(bins)
    plt.tight_layout()
    plt.savefig(
        "AngleHist_PerStep_" + dict_input_path[key][16:-4] + ".png", format="png"
    )
    plt.close()
