from tifffile import imwrite
from skimage.util import img_as_uint
import os
from os.path import dirname, basename
from scipy.ndimage import gaussian_filter
import numpy as np
import pandas as pd
from rich.progress import track
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/FL_SPT_in_condensates/FL-FUS488-ALEX-PAINT/20230928-GOX-473nm-50per-640nm-20per-20msALEX/reconstructed_seperate_channel"
os.chdir(folder_save)
lst_files = [f for f in os.listdir(".") if f.endswith("spots_reformatted.csv")]

tracklength_threshold = 10

um_per_pixel = 0.117
scaling_factor = 1
um_per_pixel_PAINT = um_per_pixel / scaling_factor
xpixels_ONI = 418
ypixels_ONI = 674
xedges = np.arange((xpixels_ONI + 1) * scaling_factor)
yedges = np.arange((ypixels_ONI + 1) * scaling_factor)

for fpath in lst_files:
    fname = basename(fpath)
    df = pd.read_csv(fname)

    ## Reconstruct PAINT image

    # single-frame spots
    df_single_frame_spots = df[df["trackID"].isna()]
    img_spots, _, _ = np.histogram2d(
        x=df_single_frame_spots["x"].to_numpy(float) * scaling_factor,
        y=df_single_frame_spots["y"].to_numpy(float) * scaling_factor,
        bins=(xedges, yedges),
    )

    lst_tracklength = []
    # tracks
    df_tracks = df[df["trackID"].notna()]
    all_trackID = df_tracks["trackID"].unique()
    lst_of_arr_x = []
    lst_of_arr_y = []
    for trackID in track(all_trackID, description=fname + ":PAINT"):
        df_current = df_tracks[df_tracks["trackID"] == trackID]
        lst_tracklength.append(df_current.shape[0])
        # for short tracks, treat as spots
        if df_current.shape[0] <= tracklength_threshold:
            lst_of_arr_x.append(df_current["x"].to_numpy(float) * scaling_factor)
            lst_of_arr_y.append(df_current["y"].to_numpy(float) * scaling_factor)
            continue
        # for long tracks, randomly pick tracklength_threshold number of spots
        else:
            chosen_idx = np.random.choice(df_current.shape[0], tracklength_threshold)
            lst_of_arr_x.append(
                df_current.iloc[chosen_idx]["x"].to_numpy(float) * scaling_factor
            )
            lst_of_arr_y.append(
                df_current.iloc[chosen_idx]["y"].to_numpy(float) * scaling_factor
            )
            continue

    img_tracks, _, _ = np.histogram2d(
        x=np.hstack(lst_of_arr_x),
        y=np.hstack(lst_of_arr_y),
        bins=(xedges, yedges),
    )

    img_PAINT = img_spots + img_tracks
    img_PAINT_final = img_as_uint(img_PAINT / img_PAINT.max())

    fname_save = (
        fname.split("-spot")[0]
        + "-threshold-"
        + str(tracklength_threshold)
        + "-PAINT.tif"
    )
    imwrite(
        fname_save,
        img_PAINT_final,
        imagej=True,
        metadata={"um_per_pixel_PAINT": um_per_pixel_PAINT},
    )
    pickle.dump(img_PAINT, open(fname_save[:-4] + "_phys_unit.p", "wb"))

    ## Reconstruct step size iamge, unit: um
    lst_mid_x = []
    lst_mid_y = []
    lst_stepsize = []
    for trackID in track(all_trackID, description=fname + ":calculate step size"):
        df_current = df_tracks[df_tracks["trackID"] == trackID]
        xs = df_current["x"].to_numpy(float)
        ys = df_current["y"].to_numpy(float)
        ts = df_current["t"].to_numpy(float)
        mid_xs = (xs[1:] + xs[:-1]) / 2
        mid_ys = (ys[1:] + ys[:-1]) / 2
        steps = (
            np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2) * um_per_pixel
        )
        lst_mid_x.extend(mid_xs)
        lst_mid_y.extend(mid_ys)
        lst_stepsize.extend(steps)

    df_all_steps = pd.DataFrame(
        {
            "mid_x": lst_mid_x,
            "mid_y": lst_mid_y,
            "stepsize": lst_stepsize,
        },
        dtype=float,
    )

    # put them in grid, calculate mean
    img_stepsize = np.zeros_like(img_PAINT)
    for x in track(
        range(img_stepsize.shape[0]), description=fname + ":mean step size image"
    ):
        for y in range(img_stepsize.shape[1]):
            df_current = df_all_steps[
                df_all_steps["mid_x"].between(x, x + 1)
                & df_all_steps["mid_y"].between(y, y + 1)
            ]
            mean_stepsize = df_current["stepsize"].mean()
            img_stepsize[x, y] = mean_stepsize
    img_stepsize_no_nan = np.nan_to_num(img_stepsize)
    img_stepsize_smoothed = gaussian_filter(img_stepsize_no_nan, 1.5)
    img_stepsize_final = img_as_uint(
        img_stepsize_smoothed / img_stepsize_smoothed.max()
    )

    fname_save = fname.split("-spot")[0] + "-MeanStepSize.tif"
    imwrite(
        fname_save,
        img_stepsize_final,
        imagej=True,
        metadata={"um_per_pixel_PAINT": um_per_pixel_PAINT},
    )
    pickle.dump(img_stepsize_smoothed, open(fname_save[:-4] + "_phys_unit.p", "wb"))
