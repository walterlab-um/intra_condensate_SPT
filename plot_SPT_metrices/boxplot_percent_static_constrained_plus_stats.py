import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

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
    "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup"
)
# Run script "barplot_N_traj_total2mobile2constrained_per_FOV_per_condition.py" to get the following csv file
df_save = pd.read_csv("N_and_Fraction_per_FOV.csv")
df_plot = df_save.melt(
    id_vars=["label"],
    value_vars=["Static Fraction", "Constrained Fraction"],
)
box_pairs = [
    ("0Dex, -, 0h", "0Dex, -, 3h"),
    ("10Dex, -, 0h", "10Dex, -, 3h"),
    ("0Dex, -, 0h", "0Dex, He, 0h"),
    ("10Dex, -, 0h", "10Dex, He, 0h"),
    ("0Dex, He, 0h", "0Dex, Ce, 0h"),
    ("0Dex, He, 0h", "0Dex, Sp, 0h"),
    ("0Dex, Ce, 0h", "0Dex, Sp, 0h"),
    ("10Dex, He, 0h", "10Dex, Ce, 0h"),
    ("10Dex, He, 0h", "10Dex, Sp, 0h"),
    ("10Dex, Ce, 0h", "10Dex, Sp, 0h"),
]


plt.figure(figsize=(8, 6), dpi=300)
data = df_plot[df_plot["variable"] == "Static Fraction"]
ax = sns.boxplot(
    data=data,
    x="label",
    y="value",
    palette=color_palette,
)
ax = sns.stripplot(
    data=data,
    x="label",
    y="value",
    color="0.4",
)
test_results = add_stat_annotation(
    ax,
    data=data,
    x="label",
    y="value",
    box_pairs=box_pairs,
    test="Mann-Whitney",
    comparisons_correction="bonferroni",
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.title("Static Fraction per FOV", weight="bold")
plt.ylabel(r"Static Fraction, $D < D_{localization \/ error}$", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("boxplot_static_fraction_plus_stats.png", format="png")
plt.close()


plt.figure(figsize=(8, 6), dpi=300)
data = df_plot[df_plot["variable"] == "Constrained Fraction"]
ax = sns.boxplot(
    data=data,
    x="label",
    y="value",
    palette=color_palette,
)
ax = sns.stripplot(
    data=data,
    x="label",
    y="value",
    color="0.4",
)
test_results = add_stat_annotation(
    ax,
    data=data,
    x="label",
    y="value",
    box_pairs=box_pairs,
    test="Mann-Whitney",
    comparisons_correction="bonferroni",
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.title("Constrained Fraction per FOV", weight="bold")
plt.ylabel(r"Constrained Fraction, $\alpha < 0.5$", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("boxplot_constrained_fraction_plus_stats.png", format="png")
plt.close()
