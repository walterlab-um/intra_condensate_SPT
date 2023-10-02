from scipy.io import savemat
import pandas as pd
import numpy as np
from tkinter import filedialog as fd

# 20 ms per frame = 1/0.02 = 50 frames per second
fs = 50.0
umperpx = 0.117
locerror = 0.025
dict_cfg = {
    "fs": fs,
    "umperpx": umperpx,
    "locerror": locerror,
}

print("Choose one reformatted track file")
path = fd.askopenfilename()
df_all = pd.read_csv(path)

df_all = df_all.astype(
    {
        "trackID": int,
        "x": float,
        "y": float,
        "t": float,
    }
)
lst_trackID = df_all["trackID"].unique()

for trackID in lst_trackID:
    df_track = df_all[df_all["trackID"] == trackID].sort_values("t")
    x = df_track["x"].to_numpy(float)
    y = df_track["y"].to_numpy(float)

    track = np.vstack([x, y])

    fpath_out = path[:-4] + "-" + str(trackID) + ".mat"

    mdict = {
        "cfg": dict_cfg,
        "track": track,
    }

    savemat(fpath_out, mdict)
