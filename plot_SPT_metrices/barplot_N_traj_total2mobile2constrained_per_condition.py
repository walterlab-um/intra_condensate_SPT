import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

pd.options.mode.chained_assignment = None  # default='warn'

color_palette = ["#ffcb05", "#00274c", "#9a3324"]
os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
# alpha component threshold for constrained diffusion
threshold_alpha = 0.25

dict_input_path = {
    "0Dex, -, 0h": "SPT_results_AIO-0Dex_noTotR_0h.csv",
    "0Dex, -, 3h": "SPT_results_AIO-0Dex_noTotR_3h.csv",
    "0Dex, He, 1h": "SPT_results_AIO-0Dex_Hela_1h.csv",
    "0Dex, Ce, 1h": "SPT_results_AIO-0Dex_Cerebral_1h.csv",
    "0Dex, Sp, 1h": "SPT_results_AIO-0Dex_Spinal_1h.csv",
    "10Dex, -, 0h": "SPT_results_AIO-10Dex_noTotR_0h.csv",
    "10Dex, -, 3h": "SPT_results_AIO-10Dex_noTotR_3h.csv",
    "10Dex, He, 0h": "SPT_results_AIO-10Dex_Hela_0h.csv",
    "10Dex, Ce, 0h": "SPT_results_AIO-10Dex_Cerebral_0h.csv",
    "10Dex, Sp, 0h": "SPT_results_AIO-10Dex_Spinal_0h.csv",
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
lst_mobile = []
lst_constrained = []
for key in track(dict_input_path.keys()):
    df_current = pd.read_csv(dict_input_path[key])
    df_current = df_current.astype(
        {"linear_fit_log10D": float, "Displacement_um": float, "alpha": float}
    )
    lst_total_size.append(df_current.shape[0])
    df_mobile_byD = df_current[df_current["linear_fit_log10D"] > log10D_low]
    df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]
    lst_mobile.append(df_mobile.shape[0])
    df_constrained = df_mobile[df_mobile["alpha"] <= threshold_alpha]
    lst_constrained.append(df_constrained.shape[0])

df_plot = pd.DataFrame(
    {
        "label": dict_input_path.keys(),
        "Total": lst_total_size,
        "Mobile": lst_mobile,
        "Constrained": lst_constrained,
    },
    dtype=object,
)
df_plot = df_plot.melt(
    id_vars=["label"],
    value_vars=["Total", "Mobile", "Constrained"],
)
df_plot = df_plot.rename(columns={"variable": "Type"})
df_plot.to_csv("barplot_N_traj_per_dataset.csv", index=False)

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
plt.savefig("barplot_N_per_dataset.png", format="png")
plt.close()
