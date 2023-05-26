import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

lst_compare_pairs = [
    ("0Dex, -, 0h", "10Dex, -, 0h"),
    ("0Dex, -, 3h", "10Dex, -, 3h"),
    ("0Dex, -, 0h", "0Dex, -, 3h"),
    ("10Dex, -, 0h", "10Dex, -, 3h"),
    ("0Dex, -, 0h", "0Dex, He, 1h"),
    ("0Dex, He, 1h", "0Dex, Ce, 1h", "0Dex, Sp, 1h"),
    ("10Dex, -, 0h", "10Dex, He, 0h"),
    ("10Dex, He, 0h", "10Dex, Ce, 0h", "10Dex, Sp, 0h"),
]

os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
# threshold 1: minimla tracklength, because too short a track is too stochastic in angle distribution
threshold_tracklength = 50
# threshold 3: D error bounds to determine static molecule
s_per_frame = 0.02
static_err = 0.016
threshold_log10D = np.log10(static_err**2 / (4 * (s_per_frame)))
# threshold 4: Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um

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
dict_color_index = {
    "0Dex, -, 0h": 0,
    "0Dex, -, 3h": 1,
    "0Dex, He, 1h": 2,
    "0Dex, Ce, 1h": 3,
    "0Dex, Sp, 1h": 4,
    "10Dex, -, 0h": 5,
    "10Dex, -, 3h": 6,
    "10Dex, He, 0h": 7,
    "10Dex, Ce, 0h": 8,
    "10Dex, Sp, 0h": 9,
}


def extract_MSD_tau(df_current_file):
    global threshold_disp, threshold_log10D, threshold_tracklength

    df_longtracks = df_current_file[df_current_file["N_steps"] >= threshold_tracklength]
    df_mobile_byD = df_longtracks[
        df_longtracks["linear_fit_log10D"] >= threshold_log10D
    ]
    df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]

    lst_MSD_arrays = []
    for array_like_string in df_mobile["list_of_MSD_um2"].to_list():
        lst_MSD_arrays.append(
            np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        )
    all_MSD_um2 = np.concatenate(lst_MSD_arrays)

    lst_tau_arrays = []
    for array_like_string in df_mobile["list_of_tau_s"].to_list():
        lst_tau_arrays.append(
            np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        )
    all_tau_s = np.concatenate(lst_tau_arrays)

    df_MSDtau = pd.DataFrame(
        {
            "tau_s": all_tau_s,
            "MSD_um2": all_MSD_um2,
        },
        dtype=float,
    )

    return df_MSDtau


for i in track(range(len(lst_compare_pairs))):
    compare_pair = lst_compare_pairs[i]
    plt.figure(figsize=(5, 4), dpi=300)
    for key in compare_pair:
        color_idx = dict_color_index[key]
        df_current_file = pd.read_csv(dict_input_path[key])
        df_MSDtau = extract_MSD_tau(df_current_file)
        df_MSDtau = df_MSDtau[df_MSDtau["tau_s"] < 1]
        sns.lineplot(
            data=df_MSDtau,
            x="tau_s",
            y="MSD_um2",
            color=color_palette[color_idx],
            label=key,
        )
    plt.xlim(0, 1)
    plt.xlabel(r"$\tau$, s", weight="bold")
    plt.ylabel(r"Ensemble MSD, $\mu$m$^2$", weight="bold")
    plt.legend(ncol=1, fontsize=11, loc=2)
    plt.tight_layout()
    plt.savefig("Ensemble_MSDtau_compare-" + str(i) + "_pleaserename.png", format="png")
    plt.close()
