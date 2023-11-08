import os
from os.path import dirname, join
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import seaborn as sns

sns.set(color_codes=True, style="white")

# plot saSPT results for all replicates in a single condition
# Must pool together, because the dataset size only fullfill saSPT's requirements when it's pooled.
fpath_data = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig2_diffusion analysis/SPOTON_results-0Dex_noTR_0hr.csv"
os.chdir(dirname(fpath_data))

D_fast_range = [0.08, 1.6]
D_med_range = [0.003, 0.08]
D_bound_range = [10 ** (-5), 0.003]

# use a red seris pallet for RNA
pallete = ["#c38e6f", "#b73015", "#db766b"]


df_spoton = pd.read_csv(fpath_data, index_col="variables")
N_tracks = int(df_spoton["values"].N_tracks)
D_fast = float(df_spoton["values"].D_fast)
D_fast_SE = float(df_spoton["values"].D_fast_SE)
D_slow = float(df_spoton["values"].D_slow)
D_slow_SE = float(df_spoton["values"].D_slow_SE)
D_static = float(df_spoton["values"].D_static)
D_static_SE = float(df_spoton["values"].D_static_SE)
F_fast = float(df_spoton["values"].F_fast)
F_fast_SE = float(df_spoton["values"].F_fast_SE)
F_slow = float(df_spoton["values"].F_slow)
F_slow_SE = float(df_spoton["values"].F_slow_SE)
F_static = float(df_spoton["values"].F_static)
F_static_SE = float(df_spoton["values"].F_static_SE)


# fraction on y, D on x
plt.figure(figsize=(5, 5), dpi=300)
plt.errorbar(
    y=F_static,
    yerr=F_static_SE,
    x=D_static,
    xerr=D_static_SE,
    color=pallete[0],
    fmt="o",
    capsize=5,
    capthick=2,
)
plt.errorbar(
    y=F_slow,
    yerr=F_slow_SE,
    x=D_slow,
    xerr=D_slow_SE,
    color=pallete[1],
    fmt="o",
    capsize=5,
    capthick=2,
)
plt.errorbar(
    y=F_fast,
    yerr=F_fast_SE,
    x=D_fast,
    xerr=D_fast_SE,
    color=pallete[2],
    fmt="o",
    capsize=5,
    capthick=2,
)
plt.text(
    0.055,
    0.57,
    "N = " + str(N_tracks),
    weight="bold",
    color="black",
)
# plt.xlim(0, 1)
# plt.ylim(0, 1)
plt.gca().set_xscale("log")
plt.xlabel(r"Apparent D, $\mu$m$^2$/s", weight="bold")
plt.ylabel("Fraction", weight="bold")
plt.tight_layout()
plt.savefig(
    "SPOTON-0Dex_noTR_0hr.png",
    format="png",
    bbox_inches="tight",
)
plt.close()
