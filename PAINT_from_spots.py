from tifffile import imwrite
from skimage.util import img_as_uint
import os
from os.path import dirname, basename
from tkinter import filedialog as fd
import numpy as np
import pandas as pd
from rich.progress import track

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
    for trackID in track(all_trackID, description=fname):
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

    fname_save = (
        fname.split("-spot")[0]
        + "-threshold-"
        + str(tracklength_threshold)
        + "-PAINT.tif"
    )

    imwrite(
        fname_save,
        img_PAINT.astype("uint8"),
        imagej=True,
        metadata={"um_per_pixel_PAINT": um_per_pixel_PAINT},
    )

    # Plot tracklength histogram
