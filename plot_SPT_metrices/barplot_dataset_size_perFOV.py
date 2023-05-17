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

lst_tag = []
lst_FOVname = []
lst_N_total = []
lst_N_mobile = []
lst_N_normal_difuse = []
lst_fraction_static = []
lst_fraction_constrained = []
for key in dict_input_path.keys():
    df_current = pd.read_csv(dict_input_path[key])
    df_current = df_current.astype({"log10D_linear": float, "alpha": float})

    for FOVname in df_current.filename.unique():
        df_currentFOV = df_current[df_current.filename == FOVname]
        df_mobile = df_currentFOV[df_currentFOV["log10D_linear"] > log10D_low]
        df_normal_difuse = df_mobile[df_mobile["alpha"] > 0.5]

        N_total = df_currentFOV.shape[0]
        N_mobile = df_mobile.shape[0]
        N_normal_difuse = df_normal_difuse.shape[0]

        fraction_static = (N_total - N_mobile) / N_total
        fraction_constrained = (N_mobile - N_normal_difuse) / N_mobile

        lst_tag.append(key)
        lst_FOVname.append(FOVname)
        lst_N_total.append(N_total)
        lst_N_mobile.append(N_mobile)
        lst_N_normal_difuse.append(N_normal_difuse)
        lst_fraction_static.append(fraction_static)
        lst_fraction_constrained.append(fraction_constrained)


df_save = pd.DataFrame(
    {
        "label": lst_tag,
        "FOVname": lst_FOVname,
        "N, Total": lst_N_total,
        "N, Mobile": lst_N_mobile,
        "N, Normal Difusion": lst_N_normal_difuse,
        "Static Fraction": lst_fraction_static,
        "Constrained Fraction": lst_fraction_constrained,
    },
    dtype=object,
)
df_save.to_csv("N_and_Fraction_per_FOV.csv", index=False)

df_plot = df_save.melt(
    id_vars=["label"],
    value_vars=["N, Total", "N, Mobile", "N, Normal Difusion"],
)
df_plot = df_plot.rename(columns={"variable": "Type"})
df_plot.to_csv("barplot_N_traj_per_FOV.csv", index=False)

plt.figure(figsize=(8, 5), dpi=300)
ax = sns.barplot(
    data=df_plot,
    x="label",
    y="value",
    hue="Type",
    palette=color_palette,
)
# plt.axhline(500, ls="--", color="gray", alpha=0.7, lw=3)
plt.title("N per FOV", weight="bold")
plt.ylabel("Number of Trajectories", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("barplot_N_per_FOV.png", format="png")
plt.close()
