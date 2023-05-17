import os
from os.path import join, basename, dirname
import numpy as np
import pandas as pd
import fastspt
import fastspt.fastSPT_tools as fastSPT_tools
import fastspt.fastSPT_plot as fastSPT_plot
from tkinter import filedialog as fd
from rich.progress import track
import warnings

warnings.filterwarnings("ignore")

print("Choose all track files for SpotON:")
lst_files = list(fd.askopenfilenames())

## Generate a dictionary of parameters
Frac_Bound = [0, 1]
D_Free = [10 ** (-4), 10 ** (0.2)]  # theoretical bounds (-2.5, 0.2)
D_Bound = [10 ** (-4), 10 ** (0.2)]
sigma_bound = [0.005, 0.1]
LB = [D_Free[0], D_Bound[0], Frac_Bound[0], sigma_bound[0]]
UB = [D_Free[1], D_Bound[1], Frac_Bound[1], sigma_bound[1]]

params = {
    "UB": UB,
    "LB": LB,
    "LocError": None,  # Manually input the localization error in um: 35 nm = 0.035 um.
    "iterations": 3,  # Manually input the desired number of fitting iterations:
    "dT": 0.02,  # Time between frames in seconds
    "dZ": 0.700,  # The axial illumination slice: measured to be roughly 700 nm
    "ModelFit": [1, 2][False],
    "fit2states": True,
    "fitSigma": True,
    "a": 0.15716,
    "b": 0.20811,
    "useZcorr": True,
}


def format_to_SpotON(df):
    lst_trackID = list(set(df.trackID))
    SpotONinput = []
    for trackID in lst_trackID:
        track = df[df.trackID == trackID]
        frame = np.array([track.t.to_numpy(dtype="uint8")])
        time = np.array([track.t.to_numpy() * 0.02])
        xy = np.array([[x * 0.117, y * 0.117] for x, y in zip(track.x, track.y)])
        SpotONinput.append((xy, time, frame))
    SpotONinput = np.array(SpotONinput, dtype=object)

    return SpotONinput


fname = []
D_free = []
D_free_SE = []
D_bound = []
D_bound_SE = []
F_bound = []
F_bound_SE = []
fit_sigma = []
fit_sigma_SE = []
for file in track(lst_files):
    df = pd.read_csv(file)
    SpotONinput = format_to_SpotON(df)
    h_test = fastspt.compute_jump_length_distribution(
        SpotONinput, CDF=True, useEntireTraj=False
    )
    HistVecJumpsCDF, JumpProbCDF, HistVecJumps, JumpProb, _ = h_test

    ## Perform the fit
    fit = fastspt.fit_jump_length_distribution(
        JumpProb, JumpProbCDF, HistVecJumps, HistVecJumpsCDF, **params
    )

    fname.append(basename(file))
    D_free.append(fit.params["D_free"].value)
    D_free_SE.append(fit.params["D_free"].stderr)  # standard error
    D_bound.append(fit.params["D_bound"].value)
    D_bound_SE.append(fit.params["D_bound"].stderr)
    F_bound.append(fit.params["F_bound"].value)
    F_bound_SE.append(fit.params["F_bound"].stderr)
    fit_sigma.append(fit.params["sigma"].value)
    fit_sigma_SE.append(fit.params["sigma"].stderr)


df_save = pd.DataFrame(
    {
        "fname": fname,
        "D_free": D_free,
        "D_free_SE": D_free_SE,
        "D_bound": D_bound,
        "D_bound_SE": D_bound_SE,
        "F_bound": F_bound,
        "F_bound_SE": F_bound_SE,
        "fit_sigma": fit_sigma,
        "fit_sigma_SE": fit_sigma_SE,
    },
    dtype=float,
)
path_save = join(dirname(lst_files[0]), "SpotON_results.csv")
df_save.to_csv(path_save, index=False)
