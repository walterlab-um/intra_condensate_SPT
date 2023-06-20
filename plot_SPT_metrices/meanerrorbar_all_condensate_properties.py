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
os.chdir(
    "/Volumes/AnalysisGG/PROCESSED_DATA/RNA-diffusion-in-FUS/RNA_SPT_in_FUS-May2023_wrapup"
)

dict_input_path = {
    "0Dex, -, 0h": "condensates_AIO-0Dex_noTotR_0h.csv",
    "0Dex, -, 3h": "condensates_AIO-0Dex_noTotR_3h.csv",
    "0Dex, He, 1h": "condensates_AIO-0Dex_Hela_1h.csv",
    "0Dex, Ce, 1h": "condensates_AIO-0Dex_Cerebral_1h.csv",
    "0Dex, Sp, 1h": "condensates_AIO-0Dex_Spinal_1h.csv",
    "10Dex, -, 0h": "condensates_AIO-10Dex_noTotR_0h.csv",
    "10Dex, -, 3h": "condensates_AIO-10Dex_noTotR_3h.csv",
    "10Dex, He, 0h": "condensates_AIO-10Dex_Hela_0h.csv",
    "10Dex, Ce, 0h": "condensates_AIO-10Dex_Cerebral_0h.csv",
    "10Dex, Sp, 0h": "condensates_AIO-10Dex_Spinal_0h.csv",
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

## Exctra per-FOV metrics and mean within each replicate
# Output file structure
columns = [
    "label",
    "replicate_prefix",
    "Mean Area, um^2",
    "Mean Radius, nm",
    "Mean Intensity",
    "Mean Aspect Ratio",
    "Mean Solidity",
    "Mean Extent",
    "Mean N per FOV",
    "Mean Volume Fraction",
]
lst_rows_of_df = []
for key in track(dict_input_path.keys()):
    df_current_file = pd.read_csv(dict_input_path[key])
    df_current_file = df_current_file.astype(
        {
            "area_um2": float,
            "R_nm": float,
            "mean_intensity": float,
            "aspect_ratio": float,
            "contour_solidity": float,
            "contour_extent": float,
        }
    )
    # all filenames within the current condition/file
    all_filenames = df_current_file["filename"].unique().tolist()
    # filename prefix for each replicate
    replicate_prefixs = np.unique([f.split("FOV")[0] for f in all_filenames])
    for prefix in replicate_prefixs:
        current_replicate_filenames = [f for f in all_filenames if prefix in f]
        df_current_replicate = df_current_file[
            df_current_file["filename"].isin(current_replicate_filenames)
        ]

        mean_area = df_current_replicate["area_um2"].mean()
        mean_R = df_current_replicate["R_nm"].mean()
        mean_intensity = df_current_replicate["mean_intensity"].mean()
        mean_aspect_ratio = df_current_replicate["aspect_ratio"].mean()
        mean_solidity = df_current_replicate["contour_solidity"].mean()
        mean_extent = df_current_replicate["contour_extent"].mean()

        # per FOV metrics
        lst_N_per_FOV = []
        lst_vol_frac = []
        for fname in df_current_replicate["filename"].unique():
            df_current_FOV = df_current_replicate[
                df_current_replicate["filename"] == fname
            ]
            lst_N_per_FOV.append(df_current_FOV.shape[0])
            lst_vol_frac.append(
                df_current_FOV["area_um2"].sum() / (418 * 674 * (0.117**2))
            )
        mean_N_per_FOV = np.mean(lst_N_per_FOV)
        mean_vol_frac = np.mean(lst_vol_frac)

        # save
        lst_rows_of_df.append(
            [
                key,
                prefix,
                mean_area,
                mean_R,
                mean_intensity,
                mean_aspect_ratio,
                mean_solidity,
                mean_extent,
                mean_N_per_FOV,
                mean_vol_frac,
            ]
        )


df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
df_save.to_csv("condensate_all_per_replicate.csv", index=False)


def mean_errobar_stats(column_name):
    global df_save, color_palette, box_pairs
    plt.figure(figsize=(4, 6), dpi=300)
    # ax = sns.violinplot(
    #     data=df_save,
    #     x="label",
    #     y=column_name,
    #     color="lightgray",
    #     linewidth=0.5,
    #     width=1,
    #     cut=0,
    #     saturation=0.3,
    #     inner=None,
    # )
    ax = sns.stripplot(
        data=df_save,
        x="label",
        y=column_name,
        color="0.7",
        size=3,
    )
    sns.pointplot(
        data=df_save,
        x="label",
        y=column_name,
        palette=color_palette,
        markers="_",
        scale=2,
        linestyles="",
        errorbar="sd",
        errwidth=2,
        capsize=0.2,
        ax=ax,
    )
    test_results = add_stat_annotation(
        ax,
        data=df_save,
        x="label",
        y=column_name,
        box_pairs=box_pairs,
        test="t-test_welch",
        comparisons_correction=None,
        text_format="star",
        loc="outside",
        verbose=2,
    )
    plt.ylabel(column_name, weight="bold")
    ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
    plt.xlabel("")
    plt.gca().set_ylim(bottom=0)
    fname_save = "Condensate_" + column_name + ".png"
    plt.tight_layout()
    plt.savefig(fname_save, format="png")
    plt.close()


mean_errobar_stats("Mean Area, um^2")
mean_errobar_stats("Mean Radius, nm")
mean_errobar_stats("Mean Intensity")
mean_errobar_stats("Mean Aspect Ratio")
mean_errobar_stats("Mean N per FOV")
mean_errobar_stats("Mean Volume Fraction")
