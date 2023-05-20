import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

folder_path = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup"
os.chdir(folder_path)

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
i = 0
for key in track(dict_input_fname.keys()):
    fname = dict_input_fname[key]
    df_MSDtau = pd.read_csv(fname, dtype=float)
    sns.lineplot(
        data=df_MSDtau, x="tau_s", y="MSD_um2", color=color_palette[i], label=key
    )
    i += 1

plt.ylim(0, 2000)
plt.xlabel(r"$\tau$, s", weight="bold")
plt.ylabel(r"Ensemble MSD, $\mu$m$^2$", weight="bold")
plt.legend(ncol=2, fontsize=11)
plt.tight_layout()
plt.savefig("Ensemble_MSDtau.png", format="png")
plt.close()
