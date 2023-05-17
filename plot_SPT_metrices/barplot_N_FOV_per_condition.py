from os.path import join, dirname, basename
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

dict_input_path = {
    "0Dex, -, 0h": "EffectiveD-alpha-alltracks_0Dex_noTotR_0h.csv",
    "0Dex, -, 3h": "EffectiveD-alpha-alltracks_0Dex_noTotR_3h.csv",
    "0Dex, He, 1h": "EffectiveD-alpha-alltracks_0Dex_Hela_1h.csv",
    "0Dex, Ce, 1h": "EffectiveD-alpha-alltracks_0Dex_Cerebral_1h.csv",
    "0Dex, Sp, 1h": "EffectiveD-alpha-alltracks_0Dex_Spinal_1h.csv",
    "10Dex, -, 0h": "EffectiveD-alpha-alltracks_10Dex_noTotR_0h.csv",
    "10Dex, -, 3h": "EffectiveD-alpha-alltracks_10Dex_noTotR_3h.csv",
    "10Dex, He, 1h": "EffectiveD-alpha-alltracks_10Dex_Hela_1h.csv",
    "10Dex, Ce, 1h": "EffectiveD-alpha-alltracks_10Dex_Cerebral_1h.csv",
    "10Dex, Sp, 1h": "EffectiveD-alpha-alltracks_10Dex_Spinal_1h.csv",
}


lst_N_FOV = []
for key in dict_input_path.keys():
    df_current = pd.read_csv(dict_input_path[key])
    lst_N_FOV.append(len(df_current["filename"].unique()))

df_plot = pd.DataFrame(
    {
        "label": dict_input_path.keys(),
        "N FOV": lst_N_FOV,
    },
    dtype=object,
)
df_plot = df_plot.astype({"N FOV": int})
df_plot.to_csv("barplot_N_FOV_per_condition.csv", index=False)

plt.figure(figsize=(8, 5), dpi=300)
ax = sns.barplot(
    data=df_plot,
    x="label",
    y="N FOV",
    palette=color_palette,
)
plt.title("N FOV per Condition", weight="bold")
plt.ylabel("Number of FOVs", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("barplot_N_FOV_per_condition.png", format="png")
plt.close()
