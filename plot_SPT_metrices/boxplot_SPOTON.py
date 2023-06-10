import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'


os.chdir(
    "/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup/SPOTON_results"
)
df_plot = pd.read_csv("SPOTON_concat_all.csv")
df_plot = df_plot[df_plot["N_tracks"] > 1000]
df_plot["log10D_fast"] = np.log10(df_plot["D_fast"])
df_plot["log10D_slow"] = np.log10(df_plot["D_slow"])
df_plot["log10D_static"] = np.log10(df_plot["D_static"])
dict_bounds = {
    "log10D_fast": [np.log10(0.08), np.log10(1.6)],
    "log10D_slow": [np.log10(0.003), np.log10(0.08)],
    "log10D_static": [-5, np.log10(0.003)],
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


def box_strip_plot(column_to_plot, ylabel):
    global df_plot, dict_color_palette, box_pairs
    plt.figure(figsize=(8, 5), dpi=300)
    ax = sns.boxplot(
        data=df_plot,
        x="key",
        y=column_to_plot,
        palette=dict_color_palette,
    )
    ax = sns.stripplot(
        data=df_plot,
        x="key",
        y=column_to_plot,
        color="0.4",
    )
    test_results = add_stat_annotation(
        ax,
        data=df_plot,
        x="key",
        y=column_to_plot,
        box_pairs=box_pairs,
        test="Mann-Whitney",
        comparisons_correction=None,
        text_format="star",
        loc="inside",
        verbose=2,
    )
    if column_to_plot.startswith("log10D"):
        low_bound, high_bound = dict_bounds[column_to_plot]
        ax.axhline(y=low_bound, ls="--", lw=3, color="gray", alpha=0.3)
        ax.axhline(y=high_bound, ls="--", lw=3, color="gray", alpha=0.3)
    plt.ylabel(ylabel, weight="bold")
    ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(column_to_plot + ".png", format="png")
    plt.close()


box_strip_plot("F_static", "Fraction Static")
box_strip_plot("F_slow", "Fraction Slow")
box_strip_plot("F_fast", "Fraction Fast")
box_strip_plot("log10D_static", r"Apparent log$_{10}$D Static")
box_strip_plot("log10D_slow", r"Apparent log$_{10}$D Slow")
box_strip_plot("log10D_fast", r"Apparent log$_{10}$D Fast")
