import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import seaborn as sns
from statannot import add_stat_annotation
from rich.progress import track

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'


os.chdir(
    "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_aging crowding Hela"
)
df_save = pd.read_csv("N_and_Fraction_per_replicate.csv")

lst_labels = [
    "0Dex, -, 0h",
    "0Dex, -, 3h",
    "0Dex, -, 6h",
    "0Dex, -, 8h",
    "0Dex, Hela, 0h",
    "10Dex, -, 0h",
    "10Dex, -, 3h",
    "10Dex, -, 6h",
    "10Dex, -, 8h",
    "10Dex, Hela, 0h",
]

df_plot = df_save[df_save["N, Total"] > 1000].melt(
    id_vars=["label"],
    value_vars=["Static Fraction", "Constrained Fraction"],
)

color_palette = [
    "#9b2226",
    "#8d2a2e",
    "#582326",
    "#333232",
]

#############################################
plt.figure(figsize=(3, 5), dpi=300)
data = df_plot[df_plot["variable"] == "Static Fraction"]
ax = sns.pointplot(
    data=data[data["label"].isin(lst_labels[0:4])],
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
    data=data[data["label"].isin(lst_labels[0:4])],
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.title("Aging, without Dextran", weight="bold")
plt.ylim(0.8, 1)
plt.ylabel(r"Static Fraction, $D < D_{localization \/ error}$", weight="bold")
ax.set_xticklabels(["0 h", "3 h", "6 h", "8 h"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=0)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_static_by_linear_D_noDex.png", format="png")
plt.close()

#############################################
plt.figure(figsize=(3, 5), dpi=300)
data = df_plot[df_plot["variable"] == "Static Fraction"]
ax = sns.pointplot(
    data=data[data["label"].isin(lst_labels[5:-1])],
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
    data=data[data["label"].isin(lst_labels[5:-1])],
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.title("Aging, 10% Dextran", weight="bold")
plt.ylim(0.8, 1)
plt.ylabel(r"Static Fraction, $D < D_{localization \/ error}$", weight="bold")
ax.set_xticklabels(["0 h", "3 h", "6 h", "8 h"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=0)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_static_by_linear_D_10Dex.png", format="png")
plt.close()

#############################################
plt.figure(figsize=(3, 5), dpi=300)
data = df_plot[df_plot["variable"] == "Constrained Fraction"]
ax = sns.pointplot(
    data=data[data["label"].isin(lst_labels[0:4])],
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
    data=data[data["label"].isin(lst_labels[0:4])],
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.title("Aging, without Dextran", weight="bold")
plt.ylim(0.2, 0.7)
plt.ylabel(r"Constrained Fraction, $\alpha < 0.5$", weight="bold")
ax.set_xticklabels(["0 h", "3 h", "6 h", "8 h"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=0)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_constrained_by_loglog_alpha_noDex.png", format="png")
plt.close()

#############################################
plt.figure(figsize=(3, 5), dpi=300)
data = df_plot[df_plot["variable"] == "Constrained Fraction"]
ax = sns.pointplot(
    data=data[data["label"].isin(lst_labels[5:-1])],
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
    data=data[data["label"].isin(lst_labels[5:-1])],
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.title("Aging, 10% Dextran", weight="bold")
plt.ylim(0.2, 0.7)
plt.ylabel(r"Constrained Fraction, $\alpha < 0.5$", weight="bold")
ax.set_xticklabels(["0 h", "3 h", "6 h", "8 h"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=0)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_constrained_by_loglog_alpha_10Dex.png", format="png")
plt.close()

#############################################
plt.figure(figsize=(3, 5), dpi=300)
data = df_plot[df_plot["variable"] == "Static Fraction"]
ax = sns.pointplot(
    data=data[
        data["label"].isin(
            [
                "0Dex, -, 0h",
                "0Dex, Hela, 0h",
                "10Dex, -, 0h",
                "10Dex, Hela, 0h",
            ]
        )
    ],
    x="label",
    y="value",
    color=color_palette[0],
    markers="_",
    scale=2,
    linestyles="",
    errorbar="sd",
    errwidth=2,
    capsize=0.2,
)
ax = sns.stripplot(
    data=data[
        data["label"].isin(
            [
                "0Dex, -, 0h",
                "0Dex, Hela, 0h",
                "10Dex, -, 0h",
                "10Dex, Hela, 0h",
            ]
        )
    ],
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.title("Effect of RNA", weight="bold")
plt.ylim(0.8, 1)
plt.ylabel(r"Static Fraction, $D < D_{localization \/ error}$", weight="bold")
ax.set_xticklabels(["-Dex,-RNA", "-Dex,+RNA", "+Dex,-RNA", "+Dex,+RNA"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_static_by_linear_D_compareRNA.png", format="png")
plt.close()

#############################################
plt.figure(figsize=(3, 5), dpi=300)
data = df_plot[df_plot["variable"] == "Constrained Fraction"]
ax = sns.pointplot(
    data=data[
        data["label"].isin(
            [
                "0Dex, -, 0h",
                "0Dex, Hela, 0h",
                "10Dex, -, 0h",
                "10Dex, Hela, 0h",
            ]
        )
    ],
    x="label",
    y="value",
    color=color_palette[0],
    markers="_",
    scale=2,
    linestyles="",
    errorbar="sd",
    errwidth=2,
    capsize=0.2,
)
ax = sns.stripplot(
    data=data[
        data["label"].isin(
            [
                "0Dex, -, 0h",
                "0Dex, Hela, 0h",
                "10Dex, -, 0h",
                "10Dex, Hela, 0h",
            ]
        )
    ],
    x="label",
    y="value",
    color="0.7",
    size=3,
)
plt.title("Effect of RNA", weight="bold")
plt.ylim(0.2, 0.7)
plt.ylabel(r"Constrained Fraction, $\alpha < 0.5$", weight="bold")
ax.set_xticklabels(["-Dex,-RNA", "-Dex,+RNA", "+Dex,-RNA", "+Dex,+RNA"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_constrained_by_loglog_alpha_compareRNA.png", format="png")
plt.close()
