import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'


os.chdir(
    "/Volumes/AnalysisGG/PROCESSED_DATA/RNA-diffusion-in-FUS/RNA_SPT_in_FUS-May2023_wrapup/saSPT_per_replicate"
)
df_plot = pd.read_csv("saSPT_F_static_log10D_peaks_all.csv")

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
dict_color_palette = {
    "0Dex, -, 0h": "#001219",
    "0Dex, -, 3h": "#005f73",
    "0Dex, He, 1h": "#0a9396",
    "0Dex, Ce, 1h": "#94d2bd",
    "0Dex, Sp, 1h": "#e9d8a6",
    "10Dex, -, 0h": "#ee9b00",
    "10Dex, -, 3h": "#ca6702",
    "10Dex, He, 0h": "#bb3e03",
    "10Dex, Ce, 0h": "#ae2012",
    "10Dex, Sp, 0h": "#9b2226",
}


##################################
# Plot F static
plt.figure(figsize=(4, 6), dpi=300)
ax = sns.pointplot(
    data=df_plot,
    x="key",
    y="F_static",
    palette=dict_color_palette,
    markers="_",
    scale=2,
    linestyles="",
    errorbar="sd",
    errwidth=2,
    capsize=0.2,
)
ax = sns.stripplot(
    data=df_plot,
    x="key",
    y="F_static",
    color="0.7",
    size=3,
)

plt.ylim(0, 1)

test_results = add_stat_annotation(
    ax,
    data=df_plot,
    x="key",
    y="F_static",
    box_pairs=box_pairs,
    test="t-test_welch",
    comparisons_correction=None,
    text_format="star",
    loc="outside",
    verbose=2,
)
plt.ylabel("Fraction Static", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_static_by_saSPT.png", format="png")
plt.close()

##################################
# Plot Peaks
plt.figure(figsize=(4, 6), dpi=300)
ax = sns.stripplot(
    data=df_plot,
    x="key",
    y="log10D_peaks",
    palette=dict_color_palette,
    size=3,
)
plt.ylabel("saSPT Peak Position", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("saSPT_peak_position.png", format="png")
plt.close()
