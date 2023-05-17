from os.path import join, dirname, basename
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="white")

pd.options.mode.chained_assignment = None  # default='warn'

color_palette = ["#ffcb05", "#00274c", "#9a3324"]
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

# calculate error bounds
um_per_pixel = 0.117
s_per_frame = 0.02
static_err = 0.016
um_per_pxl = 0.117
link_max = 3
log10D_low = np.log10(static_err**2 / (4 * (s_per_frame)))
log10D_high = np.log10((um_per_pxl * link_max) ** 2 / (4 * (s_per_frame)))

lst_total_size = []
lst_nonstatic = []
lst_normal_difuse = []
for key in dict_input_path.keys():
    df_current = pd.read_csv(dict_input_path[key])
    df_current = df_current.astype({"log10D_linear": float, "alpha": float})
    lst_total_size.append(df_current.shape[0])
    df_nonstatic = df_current[df_current["log10D_linear"] > log10D_low]
    lst_nonstatic.append(df_nonstatic.shape[0])
    df_normal = df_nonstatic[df_nonstatic["alpha"] > 0.5]
    lst_normal_difuse.append(df_normal.shape[0])

df_plot = pd.DataFrame(
    {
        "label": dict_input_path.keys(),
        "Total": lst_total_size,
        "Non-Static": lst_nonstatic,
        "Normal Diffusion": lst_normal_difuse,
    },
    dtype=object,
)
df_plot = df_plot.melt(
    id_vars=["label"],
    value_vars=["Total", "Non-Static", "Normal Diffusion"],
)
df_plot = df_plot.rename(columns={"variable": "Type"})

plt.figure(figsize=(8, 5), dpi=300)
ax = sns.barplot(
    data=df_plot,
    x="label",
    y="value",
    hue="Type",
    width=0.9,
    palette=color_palette,
)
plt.axhline(5000, ls="--", color="gray", alpha=0.7, lw=3)
plt.title("Dataset Size", weight="bold")
plt.ylabel("Number of Trajectories", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("barplot_dataset_size_3in1.png", format="png")
plt.close()
