import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
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

os.chdir(
    "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup/saSPT_pooled"
)
plot_xlim = [-1.5, 0]

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
    "0Dex, -, 0h": "saSPT_output-0Dex_noTotR_0h.csv",
    "0Dex, -, 3h": "saSPT_output-0Dex_noTotR_3h.csv",
    "0Dex, He, 1h": "saSPT_output-0Dex_Hela_1h.csv",
    "0Dex, Ce, 1h": "saSPT_output-0Dex_Cerebral_1h.csv",
    "0Dex, Sp, 1h": "saSPT_output-0Dex_Spinal_1h.csv",
    "10Dex, -, 0h": "saSPT_output-10Dex_noTotR_0h.csv",
    "10Dex, -, 3h": "saSPT_output-10Dex_noTotR_3h.csv",
    "10Dex, He, 0h": "saSPT_output-10Dex_Hela_0h.csv",
    "10Dex, Ce, 0h": "saSPT_output-10Dex_Cerebral_0h.csv",
    "10Dex, Sp, 0h": "saSPT_output-10Dex_Spinal_0h.csv",
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


def extract_log10D_density(df_current_file):
    range_D = df_current_file["diff_coef"].unique()
    log10D_density = []
    for log10D in range_D:
        df_current_log10D = df_current_file[df_current_file["diff_coef"] == log10D]
        log10D_density.append(df_current_log10D["mean_posterior_occupation"].sum())

    df_toplot = pd.DataFrame(
        {"log10D": np.log10(range_D), "Probability": log10D_density}, dtype=float
    )

    return df_toplot


for i in track(range(len(lst_compare_pairs))):
    compare_pair = lst_compare_pairs[i]
    plt.figure(figsize=(5, 4), dpi=300)
    for key in compare_pair:
        color_idx = dict_color_index[key]
        df_current_file = pd.read_csv(dict_input_path[key], dtype=float)

        df_toplot = extract_log10D_density(df_current_file)
        df_toplot = df_toplot[df_toplot["log10D"] > plot_xlim[0]]

        sns.lineplot(
            data=df_toplot,
            x="log10D",
            y="Probability",
            color=color_palette[color_idx],
            label=key,
        )

        # find peaks
        log10D = df_toplot["log10D"].to_numpy(dtype=float)
        prabability = df_toplot["Probability"].to_numpy(dtype=float)
        peaks_idx, _ = find_peaks(prabability)
        for x in log10D[peaks_idx]:
            plt.axvline(x, color=color_palette[color_idx], ls="--")

    # plt.xlim(df_toplot["log10D"].iloc[0], df_toplot["log10D"].iloc[-1])
    plt.xlim(plot_xlim[0], plot_xlim[1])
    plt.xlabel(r"log$_{10}$D, $\mu$m$^2$/s", weight="bold")
    plt.ylabel("Proportion", weight="bold")
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("saSPT_compare-" + str(i) + "_pleaserename.png", format="png")
    plt.close()
