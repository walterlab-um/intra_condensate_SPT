import os
from os.path import join, dirname
import numpy as np
import pandas as pd
import fastspt
import warnings

warnings.filterwarnings("ignore")
# Note that the AIO format has a intrisic threshold of 8 steps for each track since it calculates apparent D.
os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")
lst_fname = [f for f in os.listdir(".") if f.startswith("SPT_results_AIO")]
# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
# scalling factors
um_per_pixel = 0.117
s_per_frame = 0.02

#########################################
# SpotON parameters
Frac_Bound = [0, 1]
Frac_Fast = [0, 1]
# Following bounds are designed based on saSPT results
D_Fast = [10 ** (-2.5), 10 ** (0)]
D_Slow = [10 ** (-2.5), 10 ** (-1)]
D_Static = [10 ** (-3.5), 10 ** (-2)]
LB = [D_Fast[0], D_Slow[0], D_Static[0], Frac_Fast[0], Frac_Bound[0]]
UB = [D_Fast[1], D_Slow[1], D_Static[1], Frac_Fast[1], Frac_Bound[1]]

params = {
    "UB": UB,
    "LB": LB,
    "LocError": None,  # Manually input the localization error in um. None means estimating the localization error from the data.
    "iterations": 3,  # Manually input the desired number of fitting iterations:
    "dT": 0.02,  # Time between frames in seconds
    "dZ": 0.7,  # The axial illumination slice
    "ModelFit": [1, 2][False],
    "fit2states": False,
    "a": 0.15716,
    "b": 0.20811,
    "useZcorr": True,
}

#########################################
# Reformat and perform fitting


def reformat_for_SpotON(df_AIO):
    # following format was interpreted from SpotON package
    global threshold_disp, um_per_pixel, s_per_frame

    df_mobile = df_AIO[df_AIO["Displacement_um"] >= threshold_disp]

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


fname = lst_fname[0]
df_AIO = pd.read_csv(fname)
SpotONinput = reformat_for_SpotON(df_AIO)

h_test = fastspt.compute_jump_length_distribution(
    SpotONinput, CDF=True, useEntireTraj=False
)
HistVecJumpsCDF, JumpProbCDF, HistVecJumps, JumpProb, _ = h_test

## Perform the fit
fit = fastspt.fit_jump_length_distribution(
    JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF, **params
)


D_free = fit.params["D_free"].value
D_free_SE = fit.params["D_free"].stderr  # standard error
D_Static = fit.params["D_Static"].value
D_Static_SE = fit.params["D_Static"].stderr
F_bound = fit.params["F_bound"].value
F_bound_SE = fit.params["F_bound"].stderr
fit_sigma = fit.params["sigma"].value
fit_sigma_SE = fit.params["sigma"].stderr

df_save = pd.DataFrame(
    {
        "item": [
            "D_free",
            "D_free_SE",
            "D_Static",
            "D_Static_SE",
            "F_bound",
            "F_bound_SE",
            "fit_sigma",
            "fit_sigma_SE",
        ],
        "value": [
            D_free,
            D_free_SE,
            D_Static,
            D_Static_SE,
            F_bound,
            F_bound_SE,
            fit_sigma,
            fit_sigma_SE,
        ],
    },
    dtype=float,
)
path_save = join(dirname(lst_files[0]), "SpotON_results_pooled.csv")
df_save.to_csv(path_save, index=False)
