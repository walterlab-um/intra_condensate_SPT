import os
import scipy.stats as stats
from shapely.geometry import Point, Polygon
import numpy as np
import pickle
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'


###############################################
# Setting parameters
# path
print("Enter the time between frames (seconds):")
s_per_frame = float(input())
# s_per_frame = 2
print("Choose the RNA track csv files for processing:")
lst_files = list(fd.askopenfilenames())
# scalling factors for physical units
um_per_pixel = 0.117
# diffusion dimension. Note: This is the dimension of the measured data, not the actual movements! Although particles are doing 3D diffussion, the HILO microscopy data is a projection on 2D plane and thus should be treated as 2D diffusion!
dimension = 2
# lag times for MSD fitting (unit: frame)
lags = [1, 2, 3]


###############################################
# Functions
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


def MAIN_calc_D(df_in, um_per_pixel, s_per_frame, lags, cnt_coordinates):
    """
    Calculate diffussion coefficient from MSD
    """
    # get all track id
    trackids = df_in.trackID.unique()

    lst_trackID = []
    lst_x = []  # pixel
    lst_y = []  # pixel
    lst_lag = []  # seconds
    lst_MSD = []
    lst_D = []  # um^2/s
    lst_log10D = []
    lst_R2 = []
    lst_disp = []  # total displacement, nm
    lst_conR = []  # condensate size, radius, nm
    lst_mindistance = []  # minimal distance to boundary, nm

    # iteration through all tracks
    for id in trackids:
        df_track = df_in[df_in.trackID == id]

        # check track length
        if df_track.t.max() - df_track.t.min() < 6:  # skip short tracks
            continue

        # check if RNA is within any contours
        within_condensate = False
        for cnt in cnt_coordinates:
            mask = Polygon(cnt)
            point_RNA = Point(df_track.x.mean(), df_track.y.mean())
            if point_RNA.within(mask):
                within_condensate = True
                condensate_R = np.sqrt(4 * mask.area / np.pi) / 2 * um_per_pixel * 1000
                min_distance = mask.exterior.distance(point_RNA) * um_per_pixel * 1000
        if within_condensate is False:
            continue

        # Call the above prepared functions
        MSDs = calc_MSD_NonPhysUnit(df_track, lags)
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.array(lags, dtype=float), MSDs
        )
        if slope <= 0:  # discard tracks that diffusion model can't explain
            continue

        # MSD = 2nDt (MSD: um^2, t: s, D: um^2/s, n is dimension)
        # Therefore, slope = 2nD and D = slope/2n
        D_nonphys = float(slope) / (2 * dimension)
        D_phys = D_nonphys * (um_per_pixel ** 2 / s_per_frame)

        # calculate total displacement
        Xs = np.array(df_track.x, dtype=float)
        Ys = np.array(df_track.y, dtype=float)
        disp = (
            np.sqrt((Xs[-1] - Xs[0]) ** 2 + (Ys[-1] - Ys[0]) ** 2) * um_per_pixel * 1000
        )

        # assign values
        lst_trackID.append(id)
        lst_x.append(Xs.mean())
        lst_y.append(Ys.mean())
        lst_lag.append(np.array(lags, dtype=float) * s_per_frame)
        lst_MSD.append(MSDs)
        lst_D.append(D_phys)
        lst_log10D.append(np.log10(D_phys))
        lst_R2.append(r_value ** 2)
        lst_disp.append(disp)
        lst_conR.append(condensate_R)
        lst_mindistance.append(min_distance)

    df_out = pd.DataFrame(
        {
            "trackID": lst_trackID,
            "mean_x": lst_x,
            "mean_y": lst_y,
            "lags": lst_lag,
            "MSDs": lst_MSD,
            "D(um^2/s)": lst_D,
            "log10D": lst_log10D,
            "R2": lst_R2,
            "total displacement(nm)": lst_disp,
            "condensate R (nm)": lst_conR,
            "min distance (nm)": lst_mindistance,
        }
    )

    return df_out


###############################################
# Main body
# f = "/Volumes/AnalysisGG/PROCESSED_DATA/2022July-RNAinFUS-preliminary/20220712_FLmRNA_10FUS_1Mg_10Dex_noTotR_24C/low_freq_05Hz_FOV-RNA.csv"
for f in track(lst_files):
    df_in = pd.read_csv(f, dtype=float)
    df_in = df_in.astype({"t": int})
    fname_pkl = f.strip("-RNA.csv") + "-condensate_ilastik_contours.pkl"

    contours, img, mask = pickle.load(open(fname_pkl, "rb"))
    cnt_coordinates = []
    for cnt in contours:
        cnt_coordinates.append([tuple(cnt[i, 0]) for i in range(cnt.shape[0])])

    df_out = MAIN_calc_D(df_in, um_per_pixel, s_per_frame, lags, cnt_coordinates)
    fname_save = f.strip(".csv") + "_linregress_D.csv"
    df_out.to_csv(fname_save)
