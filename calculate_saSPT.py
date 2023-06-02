import os
from os.path import join, dirname
import numpy as np
import pandas as pd
from saspt import sample_detections, StateArray, RBME
from tkinter import filedialog as fd
from rich.progress import track


saSPT_settings = dict(
    likelihood_type=RBME,
    pixel_size_um=0.117,
    frame_interval=0.02,
    focal_depth=0.7,
    progress_bar=True,
)
os.chdir("/Volumes/AnalysisGG/PROCESSED_DATA/RNA_SPT_in_FUS-May2023_wrapup")

lst_fname = [f for f in os.listdir(".") if f.startswith("SPT_results_AIO")]

for 
df_pooled = pd.DataFrame(
    {"x": lst_x, "y": lst_y, "trajectory": lst_trackID, "frame": lst_frame,},
    dtype=object,
)
# saSPT
SA = StateArray.from_detections(df_pooled, **saSPT_settings)
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
