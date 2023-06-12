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
static_threshold_log10D = -2

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
# Output file structure
columns = [
    "condition",
    "replicate_prefix",
    "F_static",
    "log10D_peaks",
]


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


# Main
lst_rows_of_df = []
lst_keys = list(dict_condition_prefix.keys())
for key in track(lst_keys):
    # list all replicates in the current condition
    lst_fname_current_condition = [
        f
        for f in os.listdir(".")
        if f.endswith(".csv") and f.startswith(dict_condition_prefix[key])
    ]

    for fname in lst_fname_current_condition:
        df_current_file = pd.read_csv(fname, dtype=float)
        df_toplot = extract_log10D_density(df_current_file)

        # calculate F_static
        log10D = df_toplot["log10D"].to_numpy(dtype=float)
        proportion = df_toplot["Probability"].to_numpy(dtype=float)
        F_static = proportion[log10D < static_threshold_log10D].sum()

        # find peaks in only mobile fraction
        proportion_mobile = proportion[log10D >= static_threshold_log10D]
        log10D_mobile = log10D[log10D >= static_threshold_log10D]
        peaks_idx, _ = find_peaks(proportion_mobile)
        log10D_peaks = log10D_mobile[peaks_idx]

        # save
        lst_rows_of_df.append(
            [
                key,
                fname,
                F_static,
                log10D_peaks,
            ]
        )

df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
df_save.to_csv("saSPT_F_static_log10D_peaks_all.csv", index=False)
