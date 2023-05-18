import os
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

folder_path = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup"
os.chdir(folder_path)

color_palette = ["#00274c", "#9a3324", "#ffcb05"]
dict_input_fname = {
    "0Dex, -, 0h": "MSDtau-0Dex_noTotR_0h.csv",
    "0Dex, -, 3h": "MSDtau-0Dex_noTotR_3h.csv",
    "0Dex, He, 1h": "MSDtau-0Dex_Hela_1h.csv",
    "0Dex, Ce, 1h": "MSDtau-0Dex_Cerebral_1h.csv",
    "0Dex, Sp, 1h": "MSDtau-0Dex_Spinal_1h.csv",
    "10Dex, -, 0h": "MSDtau-10Dex_noTotR_0h.csv",
    "10Dex, -, 3h": "MSDtau-10Dex_noTotR_3h.csv",
    "10Dex, He, 1h": "MSDtau-10Dex_Hela_1h.csv",
    "10Dex, Ce, 1h": "MSDtau-10Dex_Cerebral_1h.csv",
    "10Dex, Sp, 1h": "MSDtau-10Dex_Spinal_1h.csv",
}

plt.figure(figsize=(5, 4), dpi=300)
# curve 1
key = "0Dex, -, 0h"
color_idx = 0
df_MSDtau = pd.read_csv(dict_input_fname[key], dtype=float)
sns.lineplot(
    data=df_MSDtau, x="tau_s", y="MSD_um2", color=color_palette[color_idx], label=key
)
# curve 2
key = "10Dex, -, 0h"
color_idx = 1
df_MSDtau = pd.read_csv(dict_input_fname[key], dtype=float)
sns.lineplot(
    data=df_MSDtau, x="tau_s", y="MSD_um2", color=color_palette[color_idx], label=key
)
# curve 3
# key = "10Dex, Sp, 1h"
# color_idx = 2
# df_MSDtau = pd.read_csv(dict_input_fname[key], dtype=float)
# sns.lineplot(
#     data=df_MSDtau, x="tau_s", y="MSD_um2", color=color_palette[color_idx], label=key
# )
# other plot parameters
plt.ylim(0, 2000)
plt.xlim(0, 4)
plt.xlabel(r"$\tau$, s", weight="bold")
plt.ylabel(r"Ensemble MSD, $\mu$m$^2$", weight="bold")
plt.legend(ncol=1, fontsize=11, loc=2)
plt.tight_layout()
plt.savefig("Ensemble_MSDtau_compare_noVs10Dex.png", format="png")
plt.close()
