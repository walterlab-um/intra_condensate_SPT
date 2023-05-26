import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
# threshold 1: minimla tracklength
threshold_tracklength = 8
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
    global last_column_name, threshold_disp, threshold_log10D, threshold_tracklength

    df_current_file = df_current_file.astype(
        {
            "linear_fit_log10D": float,
            "N_steps": int,
            "Displacement_um": float,
            last_column_name: float,
        }
    )

    df_longtracks = df_current_file[df_current_file["N_steps"] >= threshold_tracklength]
    df_mobile_byD = df_longtracks[
        df_longtracks["linear_fit_log10D"] >= threshold_log10D
    ]
    df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]

    return df_mobile


def kde_jointplot():
    global color, col_name_label_lim_x, col_name_label_lim_y, data, fname
    plt.figure(figsize=(5, 5), dpi=300)
    sns.jointplot(
        data=data,
        x=col_name_label_lim_x[0],
        y=col_name_label_lim_y[0],
        kind="kde",
        color=color,
        fill=True,
        thresh=-1e-3,
        levels=100,
        cut=0,
        clip=(col_name_label_lim_x[2], col_name_label_lim_y[2]),
        cbar=True,
    )
    title = " ".join(dict_input_path[key][:-4].split("_"))
    plt.title(title, weight="bold")
    plt.ylabel(col_name_label_lim_y[1], weight="bold")
    plt.xlabel(col_name_label_lim_x[1], weight="bold")
    plt.ylim(col_name_label_lim_y[2][0], col_name_label_lim_y[2][1])
    plt.xlim(col_name_label_lim_x[2][0], col_name_label_lim_x[2][1])
    plt.tight_layout()
    plt.savefig(fname, format="png")
    plt.close()


lst_keys = list(dict_input_path.keys())
for i in track(range(len(lst_keys))):
    key = lst_keys[i]
    color = color_palette[i]

    df_current_file = pd.read_csv(dict_input_path[key])
    last_column_name = df_current_file.keys().tolist()[-1]
    df_mobile = extract_mobile(df_current_file)

    col_name_label_lim_x = (
        last_column_name,
        "Probability of Angle within " + last_column_name,
        (0, 0.5),
    )
    col_name_label_lim_y = (
        "N_steps",
        "Number of Steps in a Trajectory",
        (threshold_tracklength, 100),
    )
    fname = "N_vs_constrained_" + dict_input_path[key][:-4] + ".png"

    selector = (
        (df_mobile[col_name_label_lim_x[0]] >= col_name_label_lim_x[2][0])
        & (df_mobile[col_name_label_lim_x[0]] <= col_name_label_lim_x[2][1])
        & (df_mobile[col_name_label_lim_y[0]] >= col_name_label_lim_y[2][0])
        & (df_mobile[col_name_label_lim_y[0]] <= col_name_label_lim_y[2][1])
    )
    data = df_mobile[selector]

    kde_jointplot()

    break
