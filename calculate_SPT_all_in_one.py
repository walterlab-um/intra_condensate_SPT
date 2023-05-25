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
lst_fpath = list(fd.askopenfilenames())
tracklength_threshold = 5
# lag times for linear MSD-tau fitting (unit: frame)
lags_linear = np.arange(1, 6, 1)
# lag times for log-log MSD-tau fitting (unit: frame)
lags_loglog = np.arange(1, 6, 1)

# Output file columns
angle_bins = np.linspace(0, 180, 7).astype(int)  # #boundaries = #bins + 1
lst_angle_fraction_titles = [
    "(" + str(angle_bins[i]) + "," + str(angle_bins[i + 1]) + "]"
    for i in range(len(angle_bins) - 1)
]
columns = [
    "filename",
    "trackID",
    "list_of_t",
    "list_of_x",
    "list_of_y",
    "N_steps",
    "Displacement_um",
    "mean_x_pxl",
    "mean_y_pxl",
    "list_of_MSD_um2",
    "list_of_tau_s",
    "linear_fit_slope",
    "linear_fit_R2",
    "linear_fit_sigma",
    "linear_fit_D_um2s",
    "linear_fit_log10D",
    "loglog_fit_R2",
    "loglog_fit_log10D",
    "alpha",
    "list_of_angles",
] + lst_angle_fraction_titles


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


def calc_angle(x, y):
    # x and y at time 0 and time 1
    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]
    # unit vectors of all steps, and step 0 and step 1
    vector = np.array([x1 - x0, y1 - y0])
    # convert to complex number to use np.angle
    vector_complex = 1j * vector[1, :]
    vector_complex += vector[0, :]
    angles_eachstep = np.angle(vector_complex, deg=True)
    angles = np.ediff1d(angles_eachstep)  # between adjacent steps
    # convert all angles to within range (0,+-180) for output
    angles[angles < -180] = angles[angles < -180] + 360
    angles[angles > 180] = angles[angles > 180] - 360

    return angles


for fpath in lst_fpath:
    df_current_file = pd.read_csv(fpath, dtype=float)
    df_current_file = df_current_file.astype({"t": int})
    fname = basename(fpath)
    lst_trackID_in_file = df_current_file.trackID.unique().tolist()

    lst_rows_of_df = []
    for trackID in track(lst_trackID_in_file, description=fname):
        df_current_track = df_current_file[df_current_file.trackID == trackID]
        tracklength = df_current_track.shape[0]
        if tracklength < tracklength_threshold:
            continue

        new_row = []
        # 1.'filename'
        new_row.append(fname)
        # 2.'trackID'
        new_row.append(trackID)
        # 3.'list_of_t'
        new_row.append(df_current_track["t"].to_list())
        # 4.'list_of_x'
        x = df_current_track["x"].to_numpy(dtype=float)
        new_row.append(df_current_track["x"].to_list())
        # 5.'list_of_y'
        y = df_current_track["y"].to_numpy(dtype=float)
        new_row.append(df_current_track["y"].to_list())
        # 6.'N_steps'
        new_row.append(tracklength)
        # 7.'Displacement_um'
        disp_um = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2) * um_per_pixel
        new_row.append(disp_um)
        # 8.'mean_x_pxl'
        new_row.append(x.mean())
        # 9.'mean_y_pxl'
        new_row.append(y.mean())
        # 10.'list_of_MSD_um2'
        lags = np.arange(1, tracklength - 2)
        lags_phys = lags * s_per_frame
        MSDs = calc_MSD_NonPhysUnit(df_current_track, lags)
        MSDs_phys = MSDs * (um_per_pixel**2)  # um^2
        new_row.append(MSDs_phys.tolist())
        # 11.'list_of_tau_s'
        new_row.append(lags_phys.tolist())
        # 12.'linear_fit_slope'

        # D formula with errors (MSD: um^2, t: s, D: um^2/s, n: dimension, R: motion blur coefficient; doi:10.1103/PhysRevE.85.061916)
        # diffusion dimension = 2. Note: This is the dimension of the measured data, not the actual movements! Although particles are doing 3D diffussion, the microscopy data is a projection on 2D plane and thus should be treated as 2D diffusion!
        # MSD = 2 n D tau + 2 n sigma^2 - 4 n R D tau, n=2, R=1/6
        # Therefore, slope = (2n-4nR)D = (8/3) D; intercept = 2 n sigma^2
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

        # 13.'linear_fit_R2'
        # 14.'linear_fit_sigma'
        # 15.'linear_fit_D_um2s'
        # 16.'linear_fit_log10D'
        # 17.'loglog_fit_R2'
        # 18.'loglog_fit_log10D'
        # 19.'alpha'
        # 20.'list_of_angles'
        # 21. angle fractions

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
