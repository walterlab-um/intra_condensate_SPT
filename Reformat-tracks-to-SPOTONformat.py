import os
from os.path import join, basename, dirname
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track

# The script pools and transform RNA tracks format to match SPOT-ON format
print("Enter the time between frames (seconds):")
s_per_frame = float(input())
print("Enter um per pixel:")
um_per_pixel = float(input())
print("Select RNA tracks csv files to be pooled and transformed:")
lst_files = list(fd.askopenfilenames())


lst_frame = []
lst_t = []
lst_x = []
lst_y = []
lst_trajectory = []
trackID_new = 0

# to monitor unusual maximum gap
lst_maxgap = []
lst_fname = []

for file in track(lst_files):
    df_in = pd.read_csv(file, dtype=float)
    lst_trackID_old = list(set(df_in.trackID))
    for trackID_old in lst_trackID_old:
        df_track = df_in[df_in.trackID == trackID_old]
        lst_frame.extend(list(df_track.t))
        lst_t.extend(list(df_track.t.to_numpy() * s_per_frame))
        lst_x.extend(list(df_track.x.to_numpy() * um_per_pixel))
        lst_y.extend(list(df_track.y.to_numpy() * um_per_pixel))
        lst_trajectory.extend(list(np.repeat(trackID_new, df_track.shape[0])))
        trackID_new += 1

        maxgap = np.max(df_track.t[1:].to_numpy() - df_track.t[:-1].to_numpy())
        lst_maxgap.extend(list(np.repeat(maxgap, df_track.shape[0])))
        lst_fname.extend(list(np.repeat(basename(file), df_track.shape[0])))


df_out = pd.DataFrame(
    {
        "frame": lst_frame,
        "t": lst_t,
        "x": lst_x,
        "y": lst_y,
        "trajectory": lst_trajectory,
        "maxgap": lst_maxgap,
        "fname": lst_fname,
    }
)
# confirm data type
df_out = df_out.astype(
    dtype={
        "frame": "int64",
        "t": "float64",
        "x": "float64",
        "y": "float64",
        "trajectory": "int64",
        "maxgap": "int64",
        "fname": "string",
    }
)
fname_save = join(
    dirname(lst_files[0]), basename(lst_files[0]).split("-")[0] + "-SPOTON.csv"
)
df_out.to_csv(fname_save, index=False)
