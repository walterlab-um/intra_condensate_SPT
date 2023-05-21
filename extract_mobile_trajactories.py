import os
from os.path import dirname, basename, join
import pandas as pd
import numpy as np
from rich.progress import track
from tkinter import filedialog as fd

print(
    "Choose a reference file (Example: EffectiveD-alpha-alltracks_10Dex_noTotR_0h.csv)"
)
path_D_alpha_alltracks = fd.askopenfilename()
print("Please confirm alltracks reference file:\n", basename(path_D_alpha_alltracks))
folder_source = fd.askdirectory()
print("Please confirm sourse folder path:\n", folder_source)
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
lst_df_segments_mobile = []

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

    # save
    df_current_file_mobile.insert(
        0, "filename", np.repeat(fname, df_current_file_mobile.shape[0])
    )
    lst_df_segments_mobile.append(
        df_current_file_mobile[["filename", "trackID", "t", "x", "y"]]
    )

# construct and save the two dataframe
df_out_mobile = pd.concat(lst_df_segments_mobile, ignore_index=True)
fname_save = "Mobile_tracks-" + basename(path_D_alpha_alltracks).split("alltracks_")[-1]
df_out_mobile.to_csv(join(folder_save, fname_save), index=False)
