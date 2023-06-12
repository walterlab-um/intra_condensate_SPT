import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

# plot all replicates in a single condition
os.chdir(
    "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup/saSPT_per_replicate"
)
plot_xlim = [-2, 0]

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
dict_condition_prefix = {
    "0Dex, -, 0h": "0Dex_noTotR_0h",
    "0Dex, -, 3h": "0Dex_noTotR_3h",
    "0Dex, He, 1h": "0Dex_Hela_1h",
    "0Dex, Ce, 1h": "0Dex_Cerebral_1h",
    "0Dex, Sp, 1h": "0Dex_Spinal_1h",
    "10Dex, -, 0h": "10Dex_noTotR_0h",
    "10Dex, -, 3h": "10Dex_noTotR_3h",
    "10Dex, He, 0h": "10Dex_Hela_0h",
    "10Dex, Ce, 0h": "10Dex_Cerebral_0h",
    "10Dex, Sp, 0h": "10Dex_Spinal_0h",
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


lst_keys = list(dict_condition_prefix.keys())

for key in track(lst_keys):
    # list all replicates in the current condition
    lst_fname_current_condition = [
        f
        for f in os.listdir(".")
        if f.endswith(".csv") and f.startswith(dict_condition_prefix[key])
    ]

    plt.figure(figsize=(5, 4), dpi=300)
    i = 0
    for fname in lst_fname_current_condition:
        df_current_file = pd.read_csv(fname, dtype=float)
        df_toplot = extract_log10D_density(df_current_file)
        df_toplot = df_toplot[df_toplot["log10D"] > plot_xlim[0]]

        sns.lineplot(
            data=df_toplot,
            x="log10D",
            y="Probability",
            color=sns.color_palette()[i],
            label=fname,
        )
        # find peaks
        log10D = df_toplot["log10D"].to_numpy(dtype=float)
        proportion = df_toplot["Probability"].to_numpy(dtype=float)
        peaks_idx, _ = find_peaks(proportion)
        for x in log10D[peaks_idx]:
            plt.plot(
                log10D[peaks_idx],
                proportion[peaks_idx],
                "*",
                color=sns.color_palette()[i],
                markersize=10,
            )
            plt.axvline(x, color=sns.color_palette()[i], ls="--", lw=1, alpha=0.3)
        i += 1

    # plt.xlim(df_toplot["log10D"].iloc[0], df_toplot["log10D"].iloc[-1])
    plt.xlim(plot_xlim[0], plot_xlim[1])
    plt.xlabel(r"log$_{10}$D, $\mu$m$^2$/s", weight="bold")
    plt.ylabel("Proportion", weight="bold")
    plt.legend(fontsize=5)
    plt.tight_layout()
    plt.savefig(
        "saSPT_per_replicate-" + dict_condition_prefix[key] + ".png", format="png"
    )
    plt.close()
