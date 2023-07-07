import os
from os.path import dirname, basename
import numpy as np
import pandas as pd
import fastspt
import warnings
from tkinter import filedialog as fd

warnings.filterwarnings("ignore")

# Calculate SpotON using AIO format input, pooling all replicates together, for a single condition.
# Update: Each AIO file is one FOV. Let user choose all replicates within the same condition. The script will calculate SpotON for all FOVs within all replicates.

# Note that the AIO format has a intrisic threshold of 8 steps for each track since it calculates apparent D.

# From the iteration results, it seems SpotON's iteration normally do not increase rate of success! It's more about the initial guess! Let's bring it down to 1?
N_iterations = 3  # desired number of fitting iterations

print("Choose the SPT_results_AIO_concat-xxxxx.csv file for one condition:")
fpath_concat = fd.askopenfilename()
df_concat = pd.read_csv(fpath_concat)
data_folder = dirname(fpath_concat)
os.chdir(data_folder)

# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
threshold_disp_switch = False
# scalling factors
um_per_pixel = 0.117
s_per_frame = 0.02

#########################################
# Output file structure
variables = [
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
D_fast_range = [0.1, 1]
D_med_range = [0.01, 0.1]
D_bound_range = [10 ** (-4), 0.01]
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


## Perform Spot ON
SpotONinput = reformat_for_SpotON(df_concat)
N_tracks = SpotONinput.shape[0]
h_test = fastspt.compute_jump_length_distribution(
    SpotONinput, CDF=True, useEntireTraj=False
)
HistVecJumpsCDF, JumpProbCDF, HistVecJumps, JumpProb, _ = h_test

# Perform the fit
localization_error_um = df_concat["linear_fit_sigma"].mean() / 1000
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

F_slow_SE = np.nan  # re run fastspt untill it's working
while np.isnan(F_slow_SE):
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

    if np.isnan(F_slow_SE):
        print("Failed! rerun!")

print("Finally succeed!!!")
# save
values = [
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
df_save = pd.DataFrame(
    {
        "variables": variables,
        "values": values,
    },
    dtype=object,
)
fname_save = "SPOTON_results-" + basename(fpath_concat).split("concat-")[-1]
df_save.to_csv(fname_save, index=False)
