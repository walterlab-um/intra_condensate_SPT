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

# # threshold 1: minimla tracklength, to avoid tracks not
# threshold_tracklength = 50
# # threshold 3: D error bounds to determine static molecule
# s_per_frame = 0.02
# static_err = 0.016
# threshold_log10D = np.log10(static_err**2 / (4 * (s_per_frame)))
# # threshold 4: Displacement threshold for non static molecules
# threshold_disp = 0.2  # unit: um

df_all = pd.read_csv(fpath_concat)
os.chdir(dirname(fpath_concat))


def extract_MSD_tau(df_current_file):
    # global threshold_disp, threshold_log10D, threshold_tracklength

    # df_longtracks = df_current_file[df_current_file["N_steps"] >= threshold_tracklength]
    # df_mobile_byD = df_longtracks[
    #     df_longtracks["linear_fit_log10D"] >= threshold_log10D
    # ]
    # df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]

    df_mobile = df_current_file

    lst_MSD_arrays = []
    for array_like_string in df_mobile["list_of_MSD_um2"].to_list():
        lst_MSD_arrays.append(
            np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        )
    all_MSD_um2 = np.concatenate(lst_MSD_arrays)

    lst_tau_arrays = []
    for array_like_string in df_mobile["list_of_tau_s"].to_list():
        lst_tau_arrays.append(
            np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        )
    all_tau_s = np.concatenate(lst_tau_arrays)

    df_MSDtau = pd.DataFrame(
        {
            "tau_s": all_tau_s,
            "MSD_um2": all_MSD_um2,
        },
        dtype=float,
    )

    return df_MSDtau, df_mobile.shape[0]


df_MSDtau, N = extract_MSD_tau(df_all)
data = df_MSDtau[df_MSDtau["tau_s"] <= 1]
plt.figure(figsize=(5, 4), dpi=300)
# plot with 95% confidence interval
sns.lineplot(data=data, x="tau_s", y="MSD_um2", color=color, lw=3)
# plot fitting result
array_tau_s = data["tau_s"].unique()
array_MSD_um2 = np.array(
    [data[data["tau_s"] == tau]["MSD_um2"].mean() for tau in array_tau_s],
    dtype=float,
)
slope_linear, intercept_linear, R_linear, P, std_err = stats.linregress(
    array_tau_s, array_MSD_um2
)
# D formula with errors (MSD: um^2, t: s, D: um^2/s, n: dimension, R: motion blur coefficient; doi:10.1103/PhysRevE.85.061916)
# diffusion dimension = 2. Note: This is the dimension of the measured data, not the actual movements! Although particles are doing 3D diffussion, the microscopy data is a projection on 2D plane and thus should be treated as 2D diffusion!
# MSD = 2 n D tau + 2 n sigma^2 - 4 n R D tau, n=2, R=1/6
# Therefore, slope = (2n-4nR)D = (8/3) D; intercept = 2 n sigma^2
D_phys_linear = slope_linear / (8 / 3)  # um^2/s

x = np.linspace(0, 1, 100)
y = intercept_linear + slope_linear * x

plt.plot(x, y, "--", color="black", lw=1)
plt.text(
    0.08,
    0.02,  # 0.055
    "N = "
    + str(N)
    + "\n"
    + "D = "
    + str(round(D_phys_linear, 4))
    + " "
    + r"$\mu$m$^2$/s"
    + "\n"
    + r"R$^2$ = "
    + str(round(R_linear**2, 4)),
    weight="bold",
    color="black",
)

plt.xlim(0, 1)
plt.xlabel(r"$\tau$, s", weight="bold")
plt.ylabel(r"Ensemble MSD, $\mu$m$^2$", weight="bold")
plt.tight_layout()
plt.savefig("Ensemble_MSDtau_all.png", format="png")
plt.close()
