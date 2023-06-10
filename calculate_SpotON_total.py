import os
import numpy as np
import pandas as pd
import fastspt
from rich.progress import track
import warnings

warnings.filterwarnings("ignore")

# Calculate SpotON based on replicate, not FOV

# Note that the AIO format has a intrisic threshold of 8 steps for each track since it calculates apparent D.
os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
lst_fname = [f for f in os.listdir(".") if f.startswith("SPT_results_AIO")]
# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
threshold_disp_switch = False
# scalling factors
um_per_pixel = 0.117
s_per_frame = 0.02

#########################################
# Output file structure
columns = [
    "replicate_prefix",
    "N_tracks",
    "D_fast",
    "D_fast_SE",
    "D_slow",
    "D_slow_SE",
    "D_static",
    "D_static_SE",
    "F_fast",
    "F_fast_SE",
    "F_slow",
    "F_slow_SE",
    "F_static",
    "F_static_SE",
    "params",
]

#########################################
# SpotON parameters
Frac_Bound_range = [0, 1]
Frac_Fast_range = [0, 1]
# Following bounds are designed based on saSPT results
D_fast_range = [0.08, 1.6]
D_med_range = [0.003, 0.08]
D_bound_range = [10 ** (-5), 0.003]
LB = [
    D_fast_range[0],
    D_med_range[0],
    D_bound_range[0],
    Frac_Fast_range[0],
    Frac_Bound_range[0],
]
UB = [
    D_fast_range[1],
    D_med_range[1],
    D_bound_range[1],
    Frac_Fast_range[1],
    Frac_Bound_range[1],
]
N_iterations = 3  # desired number of fitting iterations
dZ = 0.7  # The axial illumination slice, um


#########################################
# Reformat and perform fitting
def reformat_for_SpotON(df_in):
    # following format was interpreted from SpotON package
    global threshold_disp, um_per_pixel, s_per_frame, threshold_disp_switch

    if threshold_disp_switch:
        df_mobile = df_in[df_in["Displacement_um"] >= threshold_disp]
    else:
        df_mobile = df_in

    SpotONinput = []
    for _, row in df_mobile.iterrows():
        array_like_string = row["list_of_x"]
        array_x = np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        array_like_string = row["list_of_y"]
        array_y = np.fromstring(array_like_string[1:-1], sep=", ", dtype=float)
        array_like_string = row["list_of_t"]
        array_frame = np.fromstring(array_like_string[1:-1], sep=", ", dtype="uint8")

        frame = np.array([array_frame])
        time = np.array([array_frame * s_per_frame])
        xy = np.array(
            [[x * um_per_pixel, y * um_per_pixel] for x, y in zip(array_x, array_y)]
        )
        SpotONinput.append((xy, time, frame))

    SpotONinput = np.array(SpotONinput, dtype=object)

    return SpotONinput


for fname in track(lst_fname):
    df_AIO = pd.read_csv(fname)
    # all filenames within the current condition/file
    all_filenames = df_AIO["filename"].unique().tolist()
    # filename prefix for each replicate
    replicate_prefixs = np.unique([f.split("FOV")[0] for f in all_filenames])
    # Main
    lst_rows_of_df = []
    for prefix in replicate_prefixs:
        current_replicate_filenames = [f for f in all_filenames if prefix in f]
        df_current_replicate = df_AIO[
            df_AIO["filename"].isin(current_replicate_filenames)
        ]

        # Perform Spot ON
        SpotONinput = reformat_for_SpotON(df_current_replicate)
        N_tracks = SpotONinput.shape[0]
        h_test = fastspt.compute_jump_length_distribution(
            SpotONinput, CDF=True, useEntireTraj=False
        )
        HistVecJumpsCDF, JumpProbCDF, HistVecJumps, JumpProb, _ = h_test

        ## Perform the fit
        localization_error_um = df_current_replicate["linear_fit_sigma"].mean() / 1000
        params = {
            "UB": UB,
            "LB": LB,
            "LocError": localization_error_um,
            "iterations": N_iterations,
            "dT": s_per_frame,
            "dZ": dZ,
            "ModelFit": [1, 2][False],
            "fit2states": False,
            "a": 0.15716,
            "b": 0.20811,
            "useZcorr": True,
        }
        fit = fastspt.fit_jump_length_distribution(
            JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF, **params
        )

        # Diffusion coefficent value
        D_fast = fit.params["D_fast"].value
        D_slow = fit.params["D_med"].value
        D_static = fit.params["D_bound"].value
        # Diffusion coefficent error
        D_fast_SE = fit.params["D_fast"].stderr
        D_slow_SE = fit.params["D_med"].stderr
        D_static_SE = fit.params["D_bound"].stderr
        # Fraction value
        F_fast = fit.params["F_fast"].value
        F_static = fit.params["F_bound"].value
        F_slow = 1 - F_fast - F_static
        # Fraction error, using Subtraction Formula for error propagation
        F_fast_SE = fit.params["F_fast"].stderr
        F_static_SE = fit.params["F_bound"].stderr
        if (F_fast_SE is None) or (F_static_SE is None):
            F_slow_SE = np.nan
        else:
            F_slow_SE = np.sqrt(F_fast_SE**2 + F_static_SE**2)

        # save
        lst_rows_of_df.append(
            [
                prefix,
                N_tracks,
                D_fast,
                D_fast_SE,
                D_slow,
                D_slow_SE,
                D_static,
                D_static_SE,
                F_fast,
                F_fast_SE,
                F_slow,
                F_slow_SE,
                F_static,
                F_static_SE,
                params,
            ]
        )

    df_save = pd.DataFrame.from_records(
        lst_rows_of_df,
        columns=columns,
    )
    fname_save = "SPOTON_results-" + fname[16:]
    df_save.to_csv(fname_save, index=False)
