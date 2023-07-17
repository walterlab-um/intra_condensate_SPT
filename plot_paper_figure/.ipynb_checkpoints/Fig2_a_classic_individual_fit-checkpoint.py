import os
from os.path import dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'

color = "#9a3324"
fpath_concat = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig2_diffusion analysis/SPT_results_AIO_concat-0Dex_noTR_0hr.csv"


threshold_disp = 0.2  # unit: um
df_all = pd.read_csv(fpath_concat)
os.chdir(dirname(fpath_concat))

# calculate error bounds
# D formula with errors (MSD: um^2, t: s, D: um^2/s, n: dimension, R: motion blur coefficient; doi:10.1103/PhysRevE.85.061916)
# diffusion dimension = 2. Note: This is the dimension of the measured data, not the actual movements! Although particles are doing 3D diffussion, the microscopy data is a projection on 2D plane and thus should be treated as 2D diffusion!
# MSD = 2 n D tau + 2 n sigma^2 - 4 n R D tau, n=2, R=1/6
# MSD = (4D - 8/6 D) tau + 4 sigma^2
# MSD = 8/3 D tau + 4 sigma^2
s_per_frame = 0.02
localization_error = df_all["linear_fit_sigma"].mean() / 1000
um_per_pxl = 0.117
link_max = 3
log10D_low = np.log10((localization_error**2) / ((8 / 3) * (s_per_frame)))
log10D_high = np.log10(((um_per_pxl * link_max) ** 2) / ((8 / 3) * (s_per_frame)))


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
plt.axvline(50, ls="--", color="dimgray", alpha=0.7)
plt.xlim(0, df_all["linear_fit_sigma"].max())
plt.xlabel("Localization Error, nm", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("LocError_histo.png", format="png", bbox_inches="tight")
plt.close()

##########################################
# Mean step size (whether static molecule)
plt.figure(figsize=(5, 3), dpi=300)
ax = sns.histplot(
    data=df_all,
    x="mean_stepsize_nm",
    bins=40,
    stat="probability",
    color=color,
    kde=True,
    lw=3,
)
plt.axvspan(
    df_all["mean_stepsize_nm"].min(),
    threshold_disp,
    facecolor="dimgray",
    alpha=0.2,
    edgecolor="none",
)
plt.text(
    180,
    0.155,
    "# Molecuels = " + str(df_all.shape[0]),
    weight="bold",
    fontsize=11,
    color="black",
)
plt.xlim(0, 300)
plt.xlabel("Trajectory Mean Step Size, nm", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("Mean_stepsize_histo.png", format="png", bbox_inches="tight")
plt.close()


##########################################
# D distribution among the non contrained molecules
data = df_all[df_all["linear_fit_R2"] > 0.7]
data = data[data["mean_stepsize_nm"] > 25]
data = data[data["alpha"] > 0.5]
plt.figure(figsize=(5, 3), dpi=300)
ax = sns.histplot(
    data=data,
    x="linear_fit_log10D",
    bins=40,
    stat="probability",
    color=color,
    kde=True,
    binrange=(log10D_low - 0.5, log10D_high + 0.5),
    lw=3,
)
plt.axvspan(
    log10D_low - 0.5, log10D_low, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.axvspan(
    log10D_high, log10D_high + 0.5, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.text(
    -0.6,
    0.065,
    "# Fitted Molecules = " + str(data.shape[0]),
    weight="bold",
    fontsize=9,
    color="black",
)
plt.xlim(log10D_low - 0.5, log10D_high + 0.5)
plt.xlabel(r"log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("ApparentD_linear_histo.png", format="png", bbox_inches="tight")
plt.close()


##########################################
# Fitting R2 of all mobile molecules except alpha < 0.2 bad fitting
data = df_all[df_all["mean_stepsize_nm"] > 25]
data = data[data["alpha"] > 0.2]
plt.figure(figsize=(5, 3), dpi=300)
ax = sns.histplot(
    data=data,
    x="linear_fit_R2",
    bins=40,
    stat="probability",
    color="#333232",
    binrange=(0, 1),
    lw=3,
    label="Linear",
)
sns.histplot(
    data=data,
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
plt.xlabel(r"Fitting R$^2$", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("fitting_R2_histo.png", format="png", bbox_inches="tight")
plt.close()


##########################################
# alpha distribution
plt.figure(figsize=(5, 3), dpi=300)
data = df_all[df_all["alpha"] < 1]
data = data[data["alpha"] > 0]
data = data[data["mean_stepsize_nm"] > 25]
ax = sns.histplot(
    data=data,
    x="alpha",
    bins=40,
    stat="probability",
    color=color,
    binrange=(0, 1),
    kde=True,
    lw=3,
)
plt.axvline(0.5, ls="--", color="dimgray", alpha=0.7)
plt.text(
    0.55,
    0.06,
    "# Mobile Molecules = " + str(data.shape[0]),
    weight="bold",
    color="black",
    fontsize=9,
)
plt.xlim(0, 1)
plt.xlabel(r"$\alpha$ Componenet)", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.savefig("alpha_histo.png", format="png", bbox_inches="tight")
plt.close()

##########################################
# angle per step distribution
data = df_all[df_all["mean_stepsize_nm"] > 25]
data = data[data["alpha"] > 0.2]
lst_angle_arrays = []
for array_like_string in data["list_of_angles"].to_list():
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
    15,
    0.049,
    "# Mobile Molecules = " + str(df_angles.shape[0]),
    weight="bold",
    color="black",
    fontsize=9,
)
plt.xlabel("Angle between Two Steps, Degree", weight="bold")
plt.ylabel("Probability", weight="bold")
plt.xlim(-180, 180)
bins = np.linspace(-180, 180, 10).astype(int)
plt.xticks(bins)
plt.tight_layout()
plt.savefig("angle_histo.png", format="png", bbox_inches="tight")
plt.close()
