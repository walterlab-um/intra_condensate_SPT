import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from rich.progress import track

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'

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
# Output file structure
columns = [
    "label",
    "replicate_prefix",
    "N, Mobile",
    "N, Constrained",
    "Constrained Fraction, by Angle",
]
# construct output dataframe
lst_rows_of_df = []
for key in track(dict_input_path.keys()):
    df_current_file = pd.read_csv(dict_input_path[key])
    lst_keys = df_current_file.keys().tolist()
    df_current_file = df_current_file.astype(
        {
            "linear_fit_log10D": float,
            "N_steps": int,
            "Displacement_um": float,
            lst_keys[-1]: float,
        }
    )
    # filter out static tracks
    df_longtracks = df_current_file[df_current_file["N_steps"] >= threshold_tracklength]
    df_mobile_byD = df_longtracks[
        df_longtracks["linear_fit_log10D"] >= threshold_log10D
    ]
    df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]
    # all filenames within the current condition/file
    all_filenames = df_mobile["filename"].unique().tolist()
    # filename prefix for each replicate
    replicate_prefixs = np.unique([f.split("FOV")[0] for f in all_filenames])

    for prefix in replicate_prefixs:
        current_replicate_filenames = [f for f in all_filenames if prefix in f]
        df_current_replicate = df_mobile[
            df_mobile["filename"].isin(current_replicate_filenames)
        ]

        N_mobile = df_current_replicate.shape[0]
        N_constrained = np.sum(
            df_current_replicate[lst_keys[-1]].to_numpy()
            > threshold_last_bin_probability
        )

        fraction_constrained = N_constrained / N_mobile

        # save
        lst_rows_of_df.append(
            [
                key,
                prefix,
                N_mobile,
                N_constrained,
                fraction_constrained,
            ]
        )


df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
df_save.to_csv("N_and_Fraction_per_replicate_byAngle.csv", index=False)


plt.figure(figsize=(4, 6), dpi=300)
ax = sns.pointplot(
    data=df_save,
    x="label",
    y="Constrained Fraction, by Angle",
    palette=color_palette,
    markers="_",
    scale=2,
    linestyles="",
    errorbar="sd",
    errwidth=2,
    capsize=0.2,
)
ax = sns.stripplot(
    data=df_save,
    x="label",
    y="Constrained Fraction, by Angle",
    color="0.7",
    size=3,
)
plt.ylim(0, 1)
test_results = add_stat_annotation(
    ax,
    data=df_save,
    x="label",
    y="Constrained Fraction, by Angle",
    box_pairs=box_pairs,
    test="t-test_welch",
    comparisons_correction=None,
    text_format="star",
    loc="outside",
    verbose=2,
)
# plt.title("Constrained Fraction per FOV, by Angle", weight="bold")
plt.ylabel("Constrained Fraction, by Angle", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_static_by_Angle.png", format="png")
plt.close()
