from tifffile import imread, imwrite
import os
from os.path import dirname, basename
import cv2
from scipy.ndimage import gaussian_filter
from math import ceil
import numpy as np
import pandas as pd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_coralled by nano domains/FUS488_FL_PAINT/example_tracks-timelapsePAINT"
os.chdir(folder_save)
lst_fname = [f for f in os.listdir(".") if f.endswith("-left.csv")]

nm_per_pixel = 117
ms_per_frame = 20
tracklength_threshold = 10
output_nm_per_pixel = 100
output_ms_per_frame = 2000
integration_N_frames = 100


def spots2PAINT(df, xedges, yedges, tracklength_threshold, nm_per_pixel):
    # This function reconstruct PAINT from the whole dataframe, assuming it covers the full FOV. Therefore, it's not for individual condensates
    # single-frame spots
    df_single_frame_spots = df[df["trackID"].isna()]
    img_spots, _, _ = np.histogram2d(
        x=df_single_frame_spots["x"].to_numpy(float)
        * nm_per_pixel
        / output_nm_per_pixel,
        y=df_single_frame_spots["y"].to_numpy(float)
        * nm_per_pixel
        / output_nm_per_pixel,
        bins=(xedges, yedges),
    )

    lst_tracklength = []
    # tracks
    df_tracks = df[df["trackID"].notna()]
    all_trackID = df_tracks["trackID"].unique()
    lst_of_arr_x = []
    lst_of_arr_y = []
    for trackID in all_trackID:
        df_current = df_tracks[df_tracks["trackID"] == trackID]
        lst_tracklength.append(df_current.shape[0])
        # for short tracks, treat as spots
        if df_current.shape[0] <= tracklength_threshold:
            lst_of_arr_x.append(
                df_current["x"].to_numpy(float) * nm_per_pixel / output_nm_per_pixel
            )
            lst_of_arr_y.append(
                df_current["y"].to_numpy(float) * nm_per_pixel / output_nm_per_pixel
            )
            continue
        # for long tracks, randomly pick tracklength_threshold number of spots
        else:
            chosen_idx = np.random.choice(df_current.shape[0], tracklength_threshold)
            lst_of_arr_x.append(
                df_current.iloc[chosen_idx]["x"].to_numpy(float)
                * nm_per_pixel
                / output_nm_per_pixel
            )
            lst_of_arr_y.append(
                df_current.iloc[chosen_idx]["y"].to_numpy(float)
                * nm_per_pixel
                / output_nm_per_pixel
            )
            continue

    img_tracks, _, _ = np.histogram2d(
        x=np.hstack(lst_of_arr_x),
        y=np.hstack(lst_of_arr_y),
        bins=(xedges, yedges),
    )

    img_PAINT = img_spots + img_tracks

    return img_PAINT


for fname in lst_fname:
    df = pd.read_csv(fname)

    # establish bins grid
    x_max_nm = df.x.max() * nm_per_pixel
    y_max_nm = df.y.max() * nm_per_pixel
    xedges = np.arange(ceil(x_max_nm / output_nm_per_pixel))
    yedges = np.arange(ceil(y_max_nm / output_nm_per_pixel))

    # determine time frames
    output_N_frames = ceil(
        (df.t.max() - integration_N_frames) / (output_ms_per_frame / ms_per_frame)
    )

    # reconstruct time-lapse PAINT
    lst_img_PAINT = []
    for frame in track(range(output_N_frames), description=fname):
        t_start = frame
        t_end = t_start + integration_N_frames
        selector = (df.t >= t_start) & (df.t <= t_end)
        df_within_window = df[selector]
        img_PAINT = spots2PAINT(
            df_within_window, xedges, yedges, tracklength_threshold, nm_per_pixel
        )
        lst_img_PAINT.append(gaussian_filter(img_PAINT, 1))
    video = np.stack(lst_img_PAINT)
    imwrite(fname[:-4] + "-timelapsePAINT.tif", video, metadata={"axes": "TYX"})
