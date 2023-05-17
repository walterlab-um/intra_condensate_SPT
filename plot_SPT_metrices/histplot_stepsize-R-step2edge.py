import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

folderpath = "/Volumes/AnalysisGG/PROCESSED_DATA/2022July-RNAinFUS-preliminary/20220712_FLmRNA_10FUS_1Mg_10Dex_noTotR_24C/"
os.chdir(folderpath)
lst_fname = [
    f
    for f in os.listdir(folderpath)
    if f.endswith("RNA_steps.csv") & f.startswith("high")
]


######################################
# Prepare DataFrame
lst_stepsize = []
lst_R = []
lst_step2edge = []
lst_distnorm = []
for fname in lst_fname:
    df_in = pd.read_csv(join(folderpath, fname))
    lst_stepsize.extend(list(df_in["stepsize"]))
    lst_R.extend(list(df_in["condensate R (nm)"]))
    lst_step2edge.extend(list(df_in["step-edge distance (nm)"]))
    lst_distnorm.extend(
        list(df_in["step-edge distance (nm)"] / df_in["condensate R (nm)"])
    )

df_plot = pd.DataFrame(
    {
        "stepsize": lst_stepsize,
        "condensate R (nm)": lst_R,
        "distance to edge (nm)": lst_step2edge,
        "normalized distance": lst_distnorm,
    },
    dtype=float,
)


######################################
plt.figure(figsize=(5, 5), dpi=200)
sns.kdeplot(data=df_plot, x="stepsize", y="condensate R (nm)", fill=True)
plt.title("Stepsize-Condensate Size", fontsize=13, fontweight="bold")
plt.xlabel("Stepsize (nm)")
plt.ylabel("Condensate's Equivalent Radius (nm)")
plt.tight_layout()
fsave = "stepsize-R-kde.png"
plt.savefig(fsave, format="png")

######################################
plt.figure(figsize=(5, 5), dpi=200)
sns.kdeplot(data=df_plot, x="stepsize", y="distance to edge (nm)", fill=True)
plt.title("Stepsize-Distance to Edge", fontsize=13, fontweight="bold")
plt.xlabel("Stepsize (nm)")
plt.ylabel("Step distance to edge (nm)")
plt.tight_layout()
fsave = "stepsize-step2edge-kde.png"
plt.savefig(fsave, format="png")

######################################
plt.figure(figsize=(5, 5), dpi=200)
sns.kdeplot(data=df_plot, x="stepsize", y="normalized distance", fill=True)
plt.title("Stepsize-\nNormalized Distance to Edge", fontsize=13, fontweight="bold")
plt.xlabel("Stepsize (nm)")
plt.ylabel("Step distance to edge, normalized")
plt.tight_layout()
fsave = "stepsize-normlaized-step2edge-kde.png"
plt.savefig(fsave, format="png")

######################################
plt.figure(figsize=(9, 4), dpi=200)
g = sns.histplot(
    data=df_plot, x="stepsize", fill=True, stat="count", alpha=0.7, bins=50,
)
plt.title(
    "Step Size Distribution \n(>6, within condensates)", fontsize=13, fontweight="bold",
)
plt.xlabel("Stepsize (nm)")
plt.tight_layout()
fsave = "Stepsize Distribution.png"
plt.savefig(fsave, format="png")


######################################
plt.figure(figsize=(9, 4), dpi=200)
g = sns.histplot(
    data=df_plot, x="normalized distance", fill=True, stat="count", alpha=0.7, bins=50,
)
plt.title(
    "Step Distance to Edge, Normalized", fontsize=13, fontweight="bold",
)
plt.xlabel("Step Distance to Edge / Condensate Radius")
plt.tight_layout()
fsave = "Step Distance2Edge Distribution.png"
plt.savefig(fsave, format="png")
