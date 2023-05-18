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


lst_fname_all = df_mobile["filename"].unique().tolist()
# construct two new dataframe columns
lst_df_segments_mobile = []
lst_df_segments_constrained = []

print("Total number of files:", str(len(lst_fname_all)))
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
    lst_df_segments_mobile.append(df_current_file_mobile[["t", "x", "y"]])
    lst_df_segments_constrained.append(df_current_file_constrained[["t", "x", "y"]])

# construct and save the two dataframe
df_out_mobile = pd.concat(lst_df_segments_mobile, ignore_index=True)
fname_save = "Mobile_tracks-" + basename(path_D_alpha_alltracks).split("alltracks_")[-1]
df_out_mobile.to_csv(join(folder_save, fname_save), index=False)

df_out_constrained = pd.concat(lst_df_segments_constrained, ignore_index=True)
fname_save = (
    "Constrained_tracks-" + basename(path_D_alpha_alltracks).split("alltracks_")[-1]
)
df_out_constrained.to_csv(join(folder_save, fname_save), index=False)
