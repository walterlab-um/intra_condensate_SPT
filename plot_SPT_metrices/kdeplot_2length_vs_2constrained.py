import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
# threshold 1: minimla tracklength, because too short a track is too stochastic in angle distribution
threshold_tracklength = 50
# threshold 2: probability of an angle falls into the last bin, need a factor like 2 here to account for the stochastic nature of SM track
N_bins = 6
threshold_last_bin_probability = (1 / N_bins) * 2
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


def extract_mobile(df_current_file):
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


lst_keys = list(dict_input_path.keys())
for i in track(range(len(lst_keys))):
    key = lst_keys[i]
    df_data = pd.read_csv(dict_input_path[key])
    color = color_palette[i]

    last_column_name = df_data.keys().tolist()[-1]
    df_data = df_data.astype(
        {last_column_name: float, "N_steps": float, "Displacement_um": float}
    )
    df_data = df_data[df_data["Displacement_um"] > 0.2]

    plt.figure(figsize=(5, 5), dpi=300)
    sns.jointplot(
        data=df_data,
        x=last_column_name,
        y="N_steps",
        kind="kde",
        color=color,
        fill=True,
        thresh=0,
        levels=100,
        cut=0,
        clip=((0, 0.5), (8, 100)),
        norm=LogNorm(),
    )
    title = " ".join(dict_input_path[key][:-4].split("_"))
    plt.title(title, weight="bold")
    plt.ylabel("Number of Steps in a Trajectory", weight="bold")
    plt.xlabel("Probability of Angle within " + last_column_name, weight="bold")
    plt.ylim(8, 100)
    plt.xlim(0, 0.5)
    plt.tight_layout()
    plt.savefig(
        join(folder_save, "N_vs_constrained_" + dict_input_path[key][:-4] + ".png"),
        format="png",
    )
    plt.close()

    # plt.figure(figsize=(6, 6), dpi=300)
    # sns.jointplot(
    #     data=df_data,
    #     x=last_column_name,
    #     y="Displacement_um",
    #     kind="kde",
    #     color=color,
    #     fill=True,
    #     thresh=0,
    #     levels=100,
    #     cut=0,
    #     clip=((0, 0.5), (0.2, 0.5)),
    #     norm=LogNorm(),
    # )
    # title = " ".join(dict_input_path[key][:-4].split("_"))
    # plt.title(title, weight="bold")
    # plt.ylabel(r"Track Displacement, $\mu$m", weight="bold")
    # plt.xlabel("Probability of Angle within " + last_column_name, weight="bold")
    # plt.ylim(0.2, 0.5)
    # plt.xlim(0, 0.5)
    # plt.tight_layout()
    # plt.savefig(
    #     join(folder_save, "Disp_vs_constrained_" + dict_input_path[key][:-4] + ".png"),
    #     format="png",
    # )
    # plt.close()
