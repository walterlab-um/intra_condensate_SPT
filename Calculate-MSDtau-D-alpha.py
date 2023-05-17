import os
from os.path import join, dirname, basename
import scipy.stats as stats
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="white")

pd.options.mode.chained_assignment = None  # default='warn'

# scalling factors for physical units
print("Default scaling factors: s_per_frame = 0.02, um_per_pixel = 0.117")
# print("micron per pixel:")
# um_per_pixel = float(input())
um_per_pixel = 0.117
# print("Enter the time between frames (seconds):")
# s_per_frame = float(input())
s_per_frame = 0.02

print("Choose the RNA track csv files for processing:")
lst_files = list(fd.askopenfilenames())
# lag times for linear MSD-tau fitting (unit: frame)
lags_linear = np.arange(1, 6, 1)
# lag times for log-log MSD-tau fitting (unit: frame)
lags_loglog = np.arange(1, 6, 1)


def calc_MSD_NonPhysUnit(df_track, lags):
    df_track_sorted = df_track.sort_values("t")
    # prepare storage arrays
    MSDs = []
    # STDs = np.array([], dtype=float)

    # filling gaps
    ref_t = list(range(df_track_sorted.t.min(), df_track_sorted.t.max()))
    missing_t = list(set(ref_t) - set(df_track_sorted.t))
    if len(missing_t) > 0:
        for t in missing_t:
            complete_t = np.append(df_track_sorted.t, np.array(missing_t, dtype=int))
            complete_x = np.append(df_track_sorted.x, np.repeat(np.NaN, len(missing_t)))
            complete_y = np.append(df_track_sorted.y, np.repeat(np.NaN, len(missing_t)))
            df_complete = pd.DataFrame()
            df_complete["x"] = complete_x
            df_complete["y"] = complete_y
            df_complete["t"] = complete_t
            df_complete = df_complete.sort_values("t")

    else:
        df_complete = df_track_sorted

    # calculate MSDs corresponding to a series of lag times
    for lag in lags:
        Xs = np.array(df_complete.x, dtype=float)
        Ys = np.array(df_complete.y, dtype=float)

        SquareDisplacements = (Xs[lag:] - Xs[:-lag]) ** 2 + (Ys[lag:] - Ys[:-lag]) ** 2
        MSD = np.nanmean(SquareDisplacements)
        MSDs.append(MSD)

    MSDs = np.array(MSDs, dtype=float)
    return MSDs


# Output file 1: MSD, tau for tau in (1, tracklength-2) in each trajector
MeanSquareDisplacement = []
tau = []
# Output file 2: Apparent D from linear fit, Apparent D + anomalous exponent α from log-log fit
lst_fname = []
lst_trackID = []
lst_meanx = []  # pixel
lst_meany = []  # pixel
lst_MSD = []
lst_slope_linear = []  # to determine
lst_R2_linear = []
lst_D_linear = []  # um^2/s
lst_log10D_linear = []
lst_sigma = []  # nm
lst_R2_loglog = []
lst_log10D_loglog = []
lst_alpha = []

# loop through every track in every file
for f in track(lst_files):
    df_in = pd.read_csv(f, dtype=float)
    df_in = df_in.astype({"t": int})
    trackids = df_in.trackID.unique()
    for id in trackids:
        df_track = df_in[df_in.trackID == id]
        tracklength = df_track.shape[0]

        # For output file 1

        lags = np.arange(1, tracklength - 2)
        lags_phys = lags * s_per_frame
        MSDs = calc_MSD_NonPhysUnit(df_track, lags)
        MSDs_phys = MSDs * (um_per_pixel**2)  # um^2
        MeanSquareDisplacement.extend(MSDs_phys)
        tau.extend(lags)

        # For output file 2

        # D formula with errors (MSD: um^2, t: s, D: um^2/s, n: dimension, R: motion blur coefficient; doi:10.1103/PhysRevE.85.061916)
        # diffusion dimension = 2. Note: This is the dimension of the measured data, not the actual movements! Although particles are doing 3D diffussion, the microscopy data is a projection on 2D plane and thus should be treated as 2D diffusion!
        # MSD = 2 n D tau + 2 n sigma^2 - 4 n R D tau, n=2, R=1/6
        # Therefore, slope = (2n-4nR)D = (8/3) D; intercept = 2 n sigma^2
        if tracklength < len(lags_linear) + 3:  # >10 steps for linear
            continue

        slope_linear, intercept_linear, R_linear, P, std_err = stats.linregress(
            lags_linear * s_per_frame, MSDs_phys[: len(lags_linear)]
        )
        if (slope_linear > 0) & (intercept_linear > 0):
            D_phys_linear = slope_linear / (8 / 3)  # um^2/s
            log10D_linear = np.log10(D_phys_linear)
            sigma_phys = np.sqrt(intercept_linear / 4) * 1000  # nm
        else:
            D_phys_linear = np.NaN
            log10D_linear = np.NaN
            sigma_phys = np.NaN

        # MSD = 2 n D tau^alpha = 4 D tau^alpha
        # log(MSD) = alpha * log(tau) + log(D) + log(4)
        # Therefore, slope = alpha; intercept = log(D) + log(4)
        if tracklength < len(lags_loglog) + 3:
            R_loglog = np.NaN
            log10D_loglog = np.NaN
            alpha = np.NaN
        else:
            slope_loglog, intercept_loglog, R_loglog, P, std_err = stats.linregress(
                np.log10(lags_loglog * s_per_frame),
                np.log10(MSDs_phys[: len(lags_loglog)]),
            )
            log10D_loglog = intercept_loglog - np.log10(4)
            alpha = slope_loglog

        # basic info
        lst_fname.append(basename(f))
        lst_trackID.append(id)
        lst_MSD.append(MSDs_phys[:5])
        lst_meanx.append(df_track.x.mean())  # pixel
        lst_meany.append(df_track.y.mean())  # pixel
        # linear fit
        lst_R2_linear.append(R_linear**2)
        lst_slope_linear.append(slope_linear)  # recorded in case negative
        lst_D_linear.append(D_phys_linear)  # um^2/s
        lst_log10D_linear.append(log10D_linear)
        lst_sigma.append(sigma_phys)  # nm
        # loglog fit
        lst_R2_loglog.append(R_loglog**2)
        lst_log10D_loglog.append(log10D_loglog)
        lst_alpha.append(alpha)

data = pd.DataFrame({"MSD_um2": MeanSquareDisplacement, "tau": tau}, dtype=float)
fpath_save = join(dirname(lst_files[0]), "MSD-tau-alltracks.csv")
data.to_csv(fpath_save, index=False)

df_out = pd.DataFrame(
    {
        "filename": lst_fname,
        "trackID": lst_trackID,
        "MSD": lst_MSD,
        "mean_x": lst_meanx,
        "mean_y": lst_meany,
        "R2_linear": lst_R2_linear,
        "slope_linear": lst_slope_linear,
        "D_linear(um^2/s)": lst_D_linear,
        "log10D_linear": lst_log10D_linear,
        "sigma(nm)": lst_sigma,
        "R2_loglog": lst_R2_loglog,
        "log10D_loglog": lst_log10D_loglog,
        "alpha": lst_alpha,
    }
)
fpath_save = join(dirname(lst_files[0]), "EffectiveD-alpha-alltracks.csv")
df_out.to_csv(fpath_save, index=False)


# Plotting

# calculate error bounds
static_err = 0.016
um_per_pxl = 0.117
link_max = 3
log10D_low = np.log10(static_err**2 / (4 * (s_per_frame)))
log10D_high = np.log10((um_per_pxl * link_max) ** 2 / (4 * (s_per_frame)))
plt.figure(figsize=(9, 4), dpi=200)
sns.histplot(
    data=df_out[df_out["R2_linear"] > 0.7].log10D_linear,
    stat="count",
    bins=30,
    color=sns.color_palette()[0],
    alpha=0.5,
)
plt.axvspan(
    log10D_low - 1.5, log10D_low, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.axvspan(
    log10D_high, log10D_high + 1.5, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.xlim(log10D_low - 1.5, log10D_high + 1.5)
plt.title("log10D, linear fitting", weight="bold")
plt.xlabel("log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
fpath_save = join(dirname(lst_files[0]), "log10D_linear.png")
plt.savefig(fpath_save, format="png")
plt.close()

plt.figure(figsize=(9, 4), dpi=200)
sns.histplot(
    data=df_out[df_out["R2_linear"] > 0.7]["sigma(nm)"],
    stat="count",
    bins=30,
    color=sns.color_palette()[1],
    alpha=0.5,
)
plt.title("Localization Error Estimate, linear fitting", weight="bold")
plt.xlabel("$\u03C3$, nm", weight="bold")
plt.tight_layout()
fpath_save = join(dirname(lst_files[0]), "sigma_linear.png")
plt.savefig(fpath_save, format="png")
plt.close()

plt.figure(figsize=(9, 4), dpi=200)
sns.histplot(
    data=df_out[df_out["R2_loglog"] > 0.7].log10D_loglog,
    stat="count",
    bins=30,
    color=sns.color_palette()[2],
    alpha=0.5,
)
plt.axvspan(
    log10D_low - 1.5, log10D_low, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.axvspan(
    log10D_high, log10D_high + 1.5, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.xlim(log10D_low - 1.5, log10D_high + 1.5)
plt.title("log10D, log-log fitting", weight="bold")
plt.xlabel("log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
fpath_save = join(dirname(lst_files[0]), "log10D_loglog.png")
plt.savefig(fpath_save, format="png")
plt.close()

plt.figure(figsize=(9, 4), dpi=200)
sns.histplot(
    data=df_out[df_out["R2_loglog"] > 0.7].alpha,
    stat="count",
    bins=30,
    color=sns.color_palette()[3],
    alpha=0.5,
)
plt.title("Anomalous Exponent", weight="bold")
plt.xlabel("$\u03B1$", weight="bold")
plt.tight_layout()
fpath_save = join(dirname(lst_files[0]), "alpha.png")
plt.savefig(fpath_save, format="png")
plt.close()
