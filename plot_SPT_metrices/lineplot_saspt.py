import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
probability_upper_limit = 0.03

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


lst_keys = list(dict_input_path.keys())
i = 0

plt.figure(figsize=(5, 4), dpi=300)
for i in track(range(len(lst_keys))):
    key = lst_keys[i]
    color = color_palette[i]
    df_current_file = pd.read_csv(dict_input_path[key], dtype=float)

    df_toplot = extract_log10D_density(df_current_file)

    sns.lineplot(
        data=df_toplot, x="log10D", y="Probability", color=color_palette[i], label=key
    )

plt.xlim(df_toplot["log10D"].iloc[0], df_toplot["log10D"].iloc[-1])
plt.ylim(0, probability_upper_limit)
plt.xlabel(r"log$_{10}$D, $\mu$m$^2$/s", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.legend(ncol=2, fontsize=11)
plt.tight_layout()
plt.savefig("saSPT_all.png", format="png")
plt.close()
