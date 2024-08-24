from tifffile import imwrite
from tkinter import filedialog as fd
import os
from os.path import dirname, basename
from scipy.ndimage import gaussian_filter
import matplotlib
import numpy as np
import pandas as pd
from rich.progress import track

matplotlib.use("Agg")
pd.options.mode.chained_assignment = None  # default='warn'

print("Choose 'spots_reformatted.csv' files from single-channel SPT-PAINT experiment:")
lst_path = list(fd.askopenfilenames())

folder_save = dirname(lst_path[0])
os.chdir(folder_save)
lst_fname = [basename(f) for f in lst_path if f.endswith("spots_reformatted.csv")]

time_cutoff = 200  # remove the first 200 frames of tracking
tracklength_threshold = 10  # distinguish long versus short tracks
condensate_area_threshold = 200  # pixels
box_padding = 3  # pixels padding arround each condensate contour
sum_loc_threshold = 10  # PAINT threshold for summed PAINT signal from both channels

um_per_pixel = 0.117
scaling_factor = 1
um_per_pixel_PAINT = um_per_pixel / scaling_factor
xpixels_ONI = 418
ypixels_ONI = 674
xedges = np.arange((xpixels_ONI + 1) * scaling_factor)
yedges = np.arange((ypixels_ONI + 1) * scaling_factor)


def spots2PAINT(df):
    # This function reconstruct PAINT from the whole dataframe, assuming it covers the full FOV. Therefore, it's not for individual condensates
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
    for trackID in track(all_trackID, description="Reconstruction: PAINT"):
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

    return img_PAINT


for fname_singlechannel in lst_fname:
    print("Now processing:", fname_singlechannel.split("-spot")[0])
    ## Reconstruct PAINT image
    df_singlechannel = pd.read_csv(fname_singlechannel)
    df_singlechannel = df_singlechannel[df_singlechannel["t"] >= time_cutoff]
    img_PAINT_singlechannel = spots2PAINT(df_singlechannel)
    imwrite(
        fname_singlechannel.split("-spot")[0] + "-PAINT.tif", img_PAINT_singlechannel
    )
    # img_denoise = gaussian_filter(img_PAINT_singlechannel, sigma=1)
    # imwrite(fname_singlechannel.split("-spot")[0] + "-PAINT-smoothed.tif", img_denoise)
