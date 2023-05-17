import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="whitegrid")

lst_fname = [
    "saSPT_output_noTotR.csv",
    "saSPT_output_spinal.csv",
]
lst_tag = [
    "no Total RNA",
    "Spinal Cord Total RNA",
]
title = "saSPT-FL-10Dex-Effect of Total RNA"

plt.figure(dpi=500)
for i in range(len(lst_fname)):
    fname = lst_fname[i]
    tag = lst_tag[i]
    df = pd.read_csv(fname)
    log10D = df["log10D"].to_numpy(dtype=float)
    prob = df["Density"].to_numpy(dtype=float)

    sns.lineplot(
        x=log10D, y=prob, color=sns.color_palette()[i], label=tag, alpha=0.5, lw=3
    )

# plt.yscale("log")
plt.title(title, weight="bold")
plt.ylabel("Probability", weight="bold")
plt.xlabel("log10D, $\mu$m$^2$/s", weight="bold")
plt.tight_layout()
plt.savefig(title + ".png", format="png")
