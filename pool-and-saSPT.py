import os
from os.path import join, dirname
import numpy as np
import pandas as pd
from saspt import sample_detections, StateArray, RBME
from tkinter import filedialog as fd
from rich.progress import track
import pickle

print("Choose all track files to be pooled for saSPT:")
lst_files = list(fd.askopenfilenames())

lst_x = []
lst_y = []
lst_trackID = []
lst_frame = []
for file in track(lst_files, description="Pooling..."):
    df = pd.read_csv(file)
    lst_x.extend(df.x)
    lst_y.extend(df.y)
    if len(lst_trackID) == 0:
        lst_trackID.extend(df.trackID)
    else:
        lst_trackID.extend(np.array(df.trackID + np.max(lst_trackID) + 1, dtype=int))
    lst_frame.extend(df.t.astype(int))

df_pooled = pd.DataFrame(
    {"x": lst_x, "y": lst_y, "trajectory": lst_trackID, "frame": lst_frame,},
    dtype=object,
)
# saSPT
settings = dict(
    likelihood_type=RBME,
    pixel_size_um=0.117,
    frame_interval=0.02,
    focal_depth=0.7,
    progress_bar=True,
)
SA = StateArray.from_detections(df_pooled, **settings)
posterior_occs = SA.posterior_occs
occurance = np.sum(posterior_occs, axis=1)
diff_coefs, loc_errors = SA.parameter_values
os.chdir(dirname(lst_files[0]))
SA.plot_occupations("saSPT_output.png")

df_save = pd.DataFrame(
    {"log10D": np.log10(diff_coefs), "Density": occurance}, dtype=float
)
path_save = join(dirname(lst_files[0]), "saSPT_output.csv")
df_save.to_csv(path_save, index=False)
