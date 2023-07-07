import os
from os.path import isdir, join, dirname, basename
import shutil
import numpy as np
import pandas as pd
from saspt import StateArray, RBME
from tkinter import filedialog as fd

# calculate saSPT results using AIO format input, for each replicate.
# Update: Each AIO file is one FOV. Let user choose all replicates within the same condition. The script will calculate saSPT for all FOVs within each replicate.

"""
IMPORTANT
Please replace the "DEFAULT PARAMETER GRIDS" section of site-packages/saspt/constants.py with an exact copy of the following:

# Guoming Gao: calculate error bounds
um_per_pxl = 0.117
s_per_frame = 0.02
localization_error = 0.02
link_max = 3
t_between_frames = 0.02
log10D_low = np.log10(localization_error**2 / (4 * (s_per_frame)))
log10D_high = np.log10((um_per_pxl * link_max) ** 2 / (4 * (t_between_frames)))

DEFAULT_DIFF_COEFS = np.logspace(log10D_low, log10D_high, 100)
DEFAULT_LOC_ERRORS = np.arange(0, 0.072, 0.002)
DEFAULT_HURST_PARS = np.arange(0.05, 1.0, 0.05)
"""

saSPT_settings = dict(
    likelihood_type=RBME,
    pixel_size_um=0.117,
    frame_interval=0.02,
    focal_depth=0.7,
    progress_bar=True,
)
# AIO file folder
print("Choose all SPT_results_AIO_xxxxx.csv files within the same condition:")
lst_AIO_files = list(fd.askopenfilenames())
save_folder = dirname(lst_AIO_files[0])
os.chdir(save_folder)

# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
# Note that the AIO format has a intrisic threshold of 8 steps for each track since it calculates apparent D.


def reformat_for_saSPT(df_AIO):
    global threshold_disp

    df_mobile = df_AIO[df_AIO["Displacement_um"] >= threshold_disp]

    lst_x = []
    for array_like_string in df_mobile["list_of_x"].to_list():
        lst_x.append(np.fromstring(array_like_string[1:-1], sep=", ", dtype=float))
    all_x = np.concatenate(lst_x)

    lst_y = []
    for array_like_string in df_mobile["list_of_y"].to_list():
        lst_y.append(np.fromstring(array_like_string[1:-1], sep=", ", dtype=float))
    all_y = np.concatenate(lst_y)

    lst_frame = []
    lst_trackID = []
    trackID = 0
    for array_like_string in df_mobile["list_of_t"].to_list():
        array_frame = np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        lst_frame.append(array_frame)
        lst_trackID.append(np.ones_like(array_frame) * trackID)
        trackID += 1
    all_frame = np.concatenate(lst_frame)
    all_trackID = np.concatenate(lst_trackID)

    df_saSPT_input = pd.DataFrame(
        {
            "x": all_x,
            "y": all_y,
            "trajectory": all_trackID,
            "frame": all_frame,
        },
        dtype=float,
    )

    return df_saSPT_input


all_filenames = [basename(f) for f in lst_AIO_files]
replicate_prefixs = np.unique([f.split("FOV")[0] for f in all_filenames])
for prefix in replicate_prefixs:
    print("Now working on: ", prefix[:-1])

    current_replicate_filenames = [f for f in all_filenames if prefix in f]
    df_current_replicate = pd.concat(
        [pd.read_csv(f) for f in current_replicate_filenames]
    )

    # skip if the dataset is too small
    if df_current_replicate.shape[0] < 1000:
        print("Dataset skip: N_tracks = ", df_current_replicate.shape[0])
        continue

    df_saSPT_input = reformat_for_saSPT(df_current_replicate)

    # saSPT
    SA = StateArray.from_detections(df_saSPT_input, **saSPT_settings)
    df_save = SA.occupations_dataframe
    fname_save = "saSPT-" + prefix[:-1] + ".csv"
    df_save.to_csv(fname_save, index=False)
    SA.plot_occupations(fname_save[:-4] + ".png")
