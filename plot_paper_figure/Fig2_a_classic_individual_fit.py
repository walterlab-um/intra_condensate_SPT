import os
from os.path import dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'

color = "#00274C"
fpath_concat = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig2_diffusion analysis/SPT_results_AIO_concat-0Dex_noTR_0hr.csv"


threshold_disp = 0.2  # unit: um
df_all = pd.read_csv(fpath_concat)
os.chdir(dirname(fpath_concat))

# calculate error bounds
s_per_frame = 0.02
localization_error = df_all["linear_fit_sigma"].mean() / 1000
um_per_pxl = 0.117
link_max = 3
log10D_low = np.log10(localization_error**2 / (4 * (s_per_frame)))
log10D_high = np.log10((um_per_pxl * link_max) ** 2 / (4 * (s_per_frame)))


##########################################
# Localization error
plt.figure(figsize=(5, 3), dpi=300)
ax = sns.histplot(
    data=df_all,
    x="linear_fit_sigma",
    bins=40,
    stat="probability",
    color=color,
    kde=True,
    lw=3,
)
# plt.text(
#     1.3,
#     0.32,
#     "N = " + str(df_all.shape[0]),
#     weight="bold",
#     color=color,
# )
plt.xlim(df_all["linear_fit_sigma"].min(), df_all["linear_fit_sigma"].max())
plt.xlabel("Localization Error, nm", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("LocError_histo.png", format="png", bbox_inches="tight")
plt.close()

##########################################
# Static by displacement
plt.figure(figsize=(5, 3), dpi=300)
ax = sns.histplot(
    data=df_all,
    x="Displacement_um",
    bins=40,
    stat="probability",
    color=color,
    kde=True,
    lw=3,
)
plt.axvspan(
    df_all["Displacement_um"].min(),
    threshold_disp,
    facecolor="dimgray",
    alpha=0.2,
    edgecolor="none",
)
plt.text(
    1.3,
    0.32,
    "N = " + str(df_all.shape[0]),
    weight="bold",
    color=color,
)
plt.xlim(df_all["Displacement_um"].min(), df_all["Displacement_um"].max())
plt.xlabel(r"Trajectory Displacement, $\mu$m", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("Displacement_histo.png", format="png", bbox_inches="tight")
plt.close()

##########################################
# D distribution among the mobile molecules plus R2
df_all = df_all[df_all["Displacement_um"] > threshold_disp]
plt.figure(figsize=(5, 3), dpi=300)
ax = sns.histplot(
    data=df_all,
    x="linear_fit_log10D",
    bins=40,
    stat="probability",
    color=color,
    kde=True,
    binrange=(log10D_low - 1.5, log10D_high + 1.5),
    lw=3,
)
plt.axvspan(
    log10D_low - 1.5, log10D_low, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.axvspan(
    log10D_high, log10D_high + 1.5, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.text(
    0.6,
    0.085,
    "N = " + str(df_all.shape[0]),
    weight="bold",
    color=color,
)
plt.xlim(log10D_low - 1.5, log10D_high + 1.5)
plt.xlabel(r"log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("ApparentD_linear_histo.png", format="png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(5, 3), dpi=300)
ax = sns.histplot(
    data=df_all,
    x="linear_fit_R2",
    bins=40,
    stat="probability",
    color="#333232",
    binrange=(0, 1),
    lw=3,
    label="Linear",
)
sns.histplot(
    data=df_all,
    x="loglog_fit_R2",
    bins=40,
    stat="probability",
    color="#f7b801",
    binrange=(0, 1),
    lw=3,
    label="Log-Log",
    ax=ax,
)
plt.legend()
plt.xlim(0, 1)
plt.xlabel(r"Linear Fitting R$^2$", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("fitting_R2_histo.png", format="png", bbox_inches="tight")
plt.close()


##########################################
# alpha distribution
plt.figure(figsize=(5, 3), dpi=300)
df_alpha_possible = df_all[df_all["alpha"] < 1]
df_alpha_possible = df_alpha_possible[df_alpha_possible["alpha"] > 0]
ax = sns.histplot(
    data=df_alpha_possible,
    x="alpha",
    bins=40,
    stat="probability",
    color=color,
    binrange=(0, 1),
    kde=True,
    lw=3,
)
plt.axvspan(0, 0.5, facecolor="#333232", alpha=0.2, edgecolor="none")
plt.axvspan(0.5, 1, facecolor="#f7b801", alpha=0.2, edgecolor="none")
plt.text(
    0.8,
    0.0305,
    "N = " + str(df_alpha_possible.shape[0]),
    weight="bold",
    color=color,
)
plt.xlim(0, 1)
plt.xlabel(r"$\alpha$ Componenet)", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("alpha_histo.png", format="png", bbox_inches="tight")
plt.close()


# angle per step
lst_angle_arrays = []
for array_like_string in df_all["list_of_angles"].to_list():
    lst_angle_arrays.append(
        np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
    )
all_angles = np.concatenate(lst_angle_arrays)
df_angles = pd.DataFrame({"angle": all_angles}, dtype=float)
plt.figure(figsize=(5, 3), dpi=300)
sns.histplot(
    data=df_angles,
    x="angle",
    bins=40,
    stat="probability",
    color=color,
    binrange=(-180, 180),
    alpha=0.5,
    lw=3,
)
plt.text(
    90,
    0.046,
    "N = " + str(df_angles.shape[0]),
    weight="bold",
    color=color,
)
plt.xlabel("Angle between Two Steps, Degree", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.xlim(-180, 180)
bins = np.linspace(-180, 180, 10).astype(int)
plt.xticks(bins)
plt.tight_layout()
plt.savefig("angle_histo.png", format="png", bbox_inches="tight")
plt.close()
