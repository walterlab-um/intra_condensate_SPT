import os
from os.path import dirname, basename, join
import pandas as pd
import numpy as np
from rich.progress import track

folder_source = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/bioFUStether-10FUS-1Mg-noDex-RT/No Total RNA/20221031_0hr/tracks-reformatted"
path_D_alpha_alltracks = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup/EffectiveD-alpha-alltracks_0Dex_noTotR_0h.csv"
folder_save = dirname(path_D_alpha_alltracks)


s_per_frame = 0.02
static_err = 0.016
log10D_low = np.log10(static_err**2 / (4 * (s_per_frame)))


os.chdir(folder_source)
df_ref = pd.read_csv(path_D_alpha_alltracks)
df_ref = df_ref.astype({"log10D_linear": float, "alpha": float})
df_mobile = df_ref[df_ref["log10D_linear"] > log10D_low]
df_constrained = df_mobile[df_mobile["alpha"] <= 0.5]


lst_fname_all = df_mobile["filename"].to_list()
# construct two new dataframe columns
lst_mobile_fname = []
lst_mobile_trackID = []
lst_mobile_t = []
lst_mobile_x = []
lst_mobile_y = []
lst_constrained_fname = []
lst_constrained_trackID = []
lst_constrained_t = []
lst_constrained_x = []
lst_constrained_y = []

for fname in track(lst_fname_all):
    df_current_file = pd.read_csv(fname)
    # extract mobile tracks from current file
    set_trackID_mobile = set(
        df_mobile[df_mobile["filename"] == fname]["trackID"].unique()
    )
    df_current_file_mobile = df_current_file[
        df_current_file["trackID"].isin(set_trackID_mobile)
    ]
    # extract constrained tracks from current file
    set_trackID_constrained = set(
        df_constrained[df_constrained["filename"] == fname]["trackID"].unique()
    )
    df_current_file_constrained = df_current_file[
        df_current_file["trackID"].isin(set_trackID_constrained)
    ]
    # save
    lst_mobile_fname.extend(np.repeat(fname, df_current_file_mobile.shape[0]).tolist())
    lst_mobile_trackID.extend(df_current_file_mobile["trackID"].to_list())
    lst_mobile_t.extend(df_current_file_mobile["t"].to_list())
    lst_mobile_x.extend(df_current_file_mobile["x"].to_list())
    lst_mobile_y.extend(df_current_file_mobile["y"].to_list())
    lst_constrained_fname.extend(
        np.repeat(fname, df_current_file_constrained.shape[0]).tolist()
    )
    lst_constrained_trackID.extend(df_current_file_constrained["trackID"].to_list())
    lst_constrained_t.extend(df_current_file_constrained["t"].to_list())
    lst_constrained_x.extend(df_current_file_constrained["x"].to_list())
    lst_constrained_y.extend(df_current_file_constrained["y"].to_list())

# construct and save the two dataframe
df_out_mobile = pd.DataFrame(
    {
        "filename": lst_mobile_fname,
        "trackID": lst_mobile_trackID,
        "t": lst_mobile_t,
        "x": lst_mobile_x,
        "y": lst_mobile_y,
    },
    dtype=object,
)
fname_save = "Mobile_tracks-" + basename(path_D_alpha_alltracks).split("alltracks_")[-1]
df_out_mobile.to_csv(join(folder_save, fname_save), index=False)

df_out_constrained = pd.DataFrame(
    {
        "filename": lst_constrained_fname,
        "trackID": lst_constrained_trackID,
        "t": lst_constrained_t,
        "x": lst_constrained_x,
        "y": lst_constrained_y,
    },
    dtype=object,
)
fname_save = (
    "Constrained_tracks-" + basename(path_D_alpha_alltracks).split("alltracks_")[-1]
)
df_out_constrained.to_csv(join(folder_save, fname_save), index=False)
