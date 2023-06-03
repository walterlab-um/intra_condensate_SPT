import os
import numpy as np
import pandas as pd
from saspt import StateArray, RBME

# calculate saSPT results using AIO format input


"""
IMPORTANT
Please replace the "DEFAULT PARAMETER GRIDS" section of site-packages/saspt/constants.py with an exact copy of the following:

static_err = 0.016
um_per_pxl = 0.117
link_max = 3
t_between_frames = 0.02
log10D_low = np.log10(static_err ** 2 / (4 * (t_between_frames)))
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
os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
# Note that the AIO format has a intrisic threshold of 8 steps for each track since it calculates apparent D.
lst_fname = [f for f in os.listdir(".") if f.startswith("SPT_results_AIO")]


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


lst_log10D = []
lst_log10D_prob = []
lst_loc_error = []
lst_loc_error_prob = []
for fname in lst_fname:
    df_AIO = pd.read_csv(fname)
    df_saSPT_input = reformat_for_saSPT(df_AIO)
    print("Done reformatting: ", fname)

    # saSPT
    SA = StateArray.from_detections(df_saSPT_input, **saSPT_settings)
    df_save = SA.occupations_dataframe
    fname_save = "saSPT_output-" + fname[16:]
    df_save.to_csv(fname_save, index=False)
    SA.plot_occupations(fname_save[:-4] + ".png")
