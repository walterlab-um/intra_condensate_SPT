import os
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")

s_per_frame = 0.02
um_per_pixel = 0.117
folder_path = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup"
os.chdir(folder_path)

lst_fname = [f for f in os.listdir(folder_path) if f.startswith("Mobile_tracks")]


def calc_MSD_NonPhysUnit(df_track, lags):
    df_track_sorted = df_track.sort_values("t")
    # prepare storage arrays
    MSDs = []

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


def install_new_trackID(df_in):
    current_t = 0
    trackID = 0
    lst_newID = []
    for t_in_track in df_in["t"]:
        if t_in_track >= current_t:
            lst_newID.append(trackID)
            current_t += 1
        elif t_in_track < current_t:
            trackID += 1
            lst_newID.append(trackID)
            current_t = 1
    df_out = deepcopy(df_in)
    df_out["trackID"] = lst_newID
    return df_out


for fname in track(lst_fname):
    data = pd.read_csv(fname)
    data = data.astype({"t": int, "x": float, "y": float})
    data = install_new_trackID(data)

    # extract all MSD and tau
    lst_MSD = []
    lst_tau = []

    trackids = data.trackID.unique()
    for id in trackids:
        df_track = data[data.trackID == id]
        tracklength = df_track.shape[0]
        lags = np.arange(1, tracklength - 2)
        lags_phys = lags * s_per_frame  # s
        MSDs = calc_MSD_NonPhysUnit(df_track, lags)
        MSDs_phys = MSDs * (um_per_pixel**2)  # um^2
        lst_MSD.extend(MSDs_phys)
        lst_tau.extend(lags_phys)

    # construct a dataframe
    df_MSDtau = pd.DataFrame(
        {
            "MSD_um2": lst_MSD,
            "tau_s": lst_tau,
        },
        dtype=float,
    )
    fname_save = "MSDtau-" + fname.split("tracks-")[-1]
    df_MSDtau.to_csv(fname_save, index=False)
