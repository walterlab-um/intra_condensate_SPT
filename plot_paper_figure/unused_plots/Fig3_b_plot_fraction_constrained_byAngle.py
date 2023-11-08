import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from rich.progress import track

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'

os.chdir(
    "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_D to organization_exclude hypothesis/crowding_aging_Hela"
)
data = pd.read_csv("N_and_Fraction_per_replicate_byAngle.csv")

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

color_palette = [
    "#9b2226",
    "#8d2a2e",
    "#582326",
    "#333232",
]

#############################################
plt.figure(figsize=(3, 5), dpi=300)
ax = sns.pointplot(
    data=data[data["label"].isin(lst_labels[0:4])],
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
    data=data[data["label"].isin(lst_labels[0:4])],
    x="label",
    y="Constrained Fraction, by Angle",
    color="0.7",
    size=3,
)
plt.title("Aging, without Dextran")
plt.ylim(0, 1)
plt.ylabel("Constrained Fraction, by Angle")
ax.set_xticklabels(["0 h", "3 h", "6 h", "8 h"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=0)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_constrained_by_Angle_noDex.png", format="png")
plt.close()

#############################################
plt.figure(figsize=(3, 5), dpi=300)
ax = sns.pointplot(
    data=data[data["label"].isin(lst_labels[5:-1])],
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
    data=data[data["label"].isin(lst_labels[5:-1])],
    x="label",
    y="Constrained Fraction, by Angle",
    color="0.7",
    size=3,
)
plt.title("Aging, 10% Dextran")
plt.ylim(0, 1)
plt.ylabel("Constrained Fraction, by Angle")
ax.set_xticklabels(["0 h", "3 h", "6 h", "8 h"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=0)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_constrained_by_Angle_10Dex.png", format="png")
plt.close()

#############################################
plt.figure(figsize=(3, 5), dpi=300)
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
    y="Constrained Fraction, by Angle",
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
    y="Constrained Fraction, by Angle",
    color="0.7",
    size=3,
)
plt.title("Effect of RNA")
plt.ylim(0, 1)
plt.ylabel("Constrained Fraction, by Angle")
ax.set_xticklabels(["-Dex,-RNA", "-Dex,+RNA", "+Dex,-RNA", "+Dex,+RNA"])
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("F_constrained_by_Angle_compareRNA.png", format="png")
plt.close()
