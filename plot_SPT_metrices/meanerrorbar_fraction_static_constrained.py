import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from rich.progress import track

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'


os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")

# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
# alpha component threshold for constrained diffusion
threshold_alpha = 0.5

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
# calculate error bounds
s_per_frame = 0.02
static_err = 0.016
# log10D_low = np.log10(static_err**2 / (4 * (s_per_frame)))
log10D_low = -3

# Output file structure
columns = [
    "label",
    "replicate_prefix",
    "N, Total",
    "N, Mobile",
    "N, Constrained",
    "Static Fraction",
    "Constrained Fraction",
]
# construct output dataframe
lst_rows_of_df = []
for key in track(dict_input_path.keys()):
    df_current = pd.read_csv(dict_input_path[key])
    df_current = df_current.astype(
        {"linear_fit_log10D": float, "Displacement_um": float, "alpha": float}
    )
    # all filenames within the current condition/file
    all_filenames = df_current["filename"].unique().tolist()
    # filename prefix for each replicate
    replicate_prefixs = np.unique([f.split("FOV")[0] for f in all_filenames])

    for prefix in replicate_prefixs:
        current_replicate_filenames = [f for f in all_filenames if prefix in f]
        df_current_replicate = df_current[
            df_current["filename"].isin(current_replicate_filenames)
        ]

        df_mobile_byD = df_current_replicate[
            df_current_replicate["linear_fit_log10D"] > log10D_low
        ]
        df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]
        df_constrained = df_mobile[df_mobile["alpha"] <= threshold_alpha]

        N_total = df_current_replicate.shape[0]
        N_mobile = df_mobile.shape[0]
        N_constrained = df_constrained.shape[0]

        if N_constrained < 1:
            continue

        F_static = (N_total - N_mobile) / N_total
        F_constrained = N_constrained / N_mobile

        # save
        lst_rows_of_df.append(
            [
                key,
                prefix,
                N_total,
                N_mobile,
                N_constrained,
                F_static,
                F_constrained,
            ]
        )

df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
df_save.to_csv("N_and_Fraction_per_replicate.csv", index=False)

df_plot = df_save[df_save["N, Total"] > 1000].melt(
    id_vars=["label"],
    value_vars=["Static Fraction", "Constrained Fraction"],
)
box_pairs = [
    ("0Dex, -, 0h", "0Dex, -, 3h"),
    ("10Dex, -, 0h", "10Dex, -, 3h"),
    ("0Dex, -, 0h", "0Dex, He, 1h"),
    ("10Dex, -, 0h", "10Dex, He, 0h"),
    ("0Dex, He, 1h", "0Dex, Ce, 1h"),
    ("0Dex, He, 1h", "0Dex, Sp, 1h"),
    ("0Dex, Ce, 1h", "0Dex, Sp, 1h"),
    ("10Dex, He, 0h", "10Dex, Ce, 0h"),
    ("10Dex, He, 0h", "10Dex, Sp, 0h"),
    ("10Dex, Ce, 0h", "10Dex, Sp, 0h"),
]

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
plt.figure(figsize=(4, 6), dpi=300)
data = df_plot[df_plot["variable"] == "Static Fraction"]
ax = sns.pointplot(
    data=data,
    x="label",
    y="value",
    palette=color_palette,
    markers="_",
    scale=2,
    linestyles="",
    errorbar="sd",
    errwidth=2,
    capsize=0.2,
)
ax = sns.stripplot(
    data=data,
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.ylim(0, 1)
test_results = add_stat_annotation(
    ax,
    data=data,
    x="label",
    y="value",
    box_pairs=box_pairs,
    test="t-test_welch",
    comparisons_correction=None,
    text_format="star",
    loc="outside",
    verbose=2,
)
# plt.title("Static Fraction per FOV", weight="bold")
plt.ylabel(r"Static Fraction, $D < D_{localization \/ error}$", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_static_by_linear_D.png", format="png")
plt.close()


plt.figure(figsize=(4, 6), dpi=300)
data = df_plot[df_plot["variable"] == "Constrained Fraction"]
ax = sns.pointplot(
    data=data,
    x="label",
    y="value",
    palette=color_palette,
    markers="_",
    scale=2,
    linestyles="",
    errorbar="sd",
    errwidth=2,
    capsize=0.2,
)
ax = sns.stripplot(
    data=data,
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.ylim(0, 1)
test_results = add_stat_annotation(
    ax,
    data=data,
    x="label",
    y="value",
    box_pairs=box_pairs,
    test="t-test_welch",
    comparisons_correction=None,
    text_format="star",
    loc="outside",
    verbose=2,
)
# plt.title("Constrained Fraction per FOV", weight="bold")
plt.ylabel(r"Constrained Fraction, $\alpha < 0.5$", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_constrained_by_loglog_alpha.png", format="png")
plt.close()
