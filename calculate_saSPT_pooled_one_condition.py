import os
from os.path import dirname, basename
import numpy as np
import pandas as pd
from saspt import StateArray, RBME
from tkinter import filedialog as fd

# calculate saSPT results using AIO format input, pooling all replicates together.
# Update: Each AIO file is one FOV. Let user choose all replicates within the same condition. The script will calculate saSPT for all FOVs within each replicate.

"""
IMPORTANT
Remember to change the "DEFAULT PARAMETER GRIDS" section of site-packages/saspt/constants.py!
"""


print("Please enter the s per frame for the video")
s_per_frame = float(input())

saSPT_settings = dict(
    likelihood_type=RBME,
    pixel_size_um=0.117,
    frame_interval=s_per_frame,
    focal_depth=0.7,
    progress_bar=True,
)

print("Choose the SPT_results_AIO_single_condition-xxxxx.csv file for one condition:")
fpath_single_condition = fd.askopenfilename()
df_single_condition = pd.read_csv(fpath_single_condition)
data_folder = dirname(fpath_single_condition)
os.chdir(data_folder)

# Note that the AIO format has a intrisic threshold of 8 steps for each track since it calculates apparent D.


def reformat_for_saSPT(df_AIO):
    lst_x = []
    for array_like_string in df_AIO["list_of_x"].to_list():
        lst_x.append(np.fromstring(array_like_string[1:-1], sep=", ", dtype=float))
    all_x = np.concatenate(lst_x)

    lst_y = []
    for array_like_string in df_AIO["list_of_y"].to_list():
        lst_y.append(np.fromstring(array_like_string[1:-1], sep=", ", dtype=float))
    all_y = np.concatenate(lst_y)

    lst_frame = []
    lst_trackID = []
    trackID = 0
    for array_like_string in df_AIO["list_of_t"].to_list():
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


# filter out immobile
df_mobile = df_single_condition[df_single_condition["mean_stepsize_nm"] > 30]
# filter out confined molecules
df_normal = df_mobile[df_mobile["alpha"] > 0.7]  # default: 0.7
df_saSPT_input = reformat_for_saSPT(df_normal)
# saSPT
SA = StateArray.from_detections(df_saSPT_input, **saSPT_settings)
df_save = SA.occupations_dataframe
fname_save = (
    "saSPT-pooled-mobile-" + basename(fpath_single_condition).split("concat-")[-1]
)
df_save.to_csv(fname_save, index=False)
# SA.plot_occupations(fname_save[:-4] + ".png")
