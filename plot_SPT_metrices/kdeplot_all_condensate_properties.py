import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# concat all csv files
lst_df = []
for key in dict_input_path.keys():
    df = pd.read_csv(dict_input_path[key])
    df.insert(0, "key", np.repeat(key, df.shape[0]))
    lst_df.append(df)

df_all = pd.concat(lst_df)
df_all.to_csv("condensate_concat_all.csv", index=False)


def hist_lineplot(column_name):
    global df_all, color_palette, box_pairs
    plt.figure(figsize=(6, 4), dpi=300)
    # add cut off by 0.95 quantile
    quantile = np.quantile(df_all[column_name], 0.95)
    ax = sns.kdeplot(
        data=df_all,
        hue="key",
        x=column_name,
        palette=color_palette,
        common_norm=False,
        clip=(0, quantile),
    )
    plt.xlabel(column_name, weight="bold")
    plt.ylabel("Density", weight="bold")
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(0.5, 1),
        ncol=5,
        frameon=False,
        title=None,
        fontsize=7,
    )
    fname_save = "Condensate_" + column_name + "_kde.png"
    plt.savefig(fname_save, format="png", bbox_inches="tight")
    plt.close()


hist_lineplot("R_nm")
hist_lineplot("mean_intensity")
