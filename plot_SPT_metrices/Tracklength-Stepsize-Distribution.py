import os
from os.path import join, dirname, basename
import numpy as np
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="white")

###############################################
# Setting parameters
print("Choose the RNA track csv files to inspect tracklength and stepsize:")
lst_files = list(fd.askopenfilenames())
folder_save = dirname(lst_files[0])

# print("Choose a saving directory:")
# folder_save = fd.askdirectory()
# scalling factors for physical units
nm_per_pixel = 117


###############################################
# Main body
tracklength = []
stepsize = []

filename = []
Ntrack = []
static_percent = []
for f in track(lst_files):
    df_in = pd.read_csv(f, dtype=float)
    df_in = df_in.astype({"t": int})
    trackID = list(set(df_in.trackID))

    tracklength_inFOV = []
    for idx in trackID:
        track = df_in[df_in.trackID == idx]
        tracklength.append(track.shape[0])
        tracklength_inFOV.append(track.shape[0])
        track = track.sort_values("t")
        phys_x = np.array(track.x * nm_per_pixel)
        phys_y = np.array(track.y * nm_per_pixel)
        phys_jump = np.sqrt(
            (phys_x[1:] - phys_x[:-1]) ** 2 + (phys_y[1:] - phys_y[:-1]) ** 2
        )
        stepsize.extend(phys_jump)

    filename.append(basename(f))
    Ntrack.append(len(trackID))
    unique, counts = np.unique(tracklength_inFOV, return_counts=True)
    static_percent.append(counts[-1] / np.sum(counts))

df_stats = pd.DataFrame(
    {"filename": filename, "Ntrack": Ntrack, "max_length_percent": static_percent},
    dtype=object,
)
fpath_save = join(folder_save, "Statistics_per_file.csv")
df_stats.to_csv(fpath_save, index=False)

plt.figure(figsize=(9, 4), dpi=200)
sns.histplot(data=tracklength, stat="count", bins=50, color="indigo", alpha=0.5)
plt.title("Tracklength Distribution")
plt.xlabel("Number of Frames")
plt.tight_layout()
fpath_save = join(folder_save, "TracklengthDistribution.png")
plt.savefig(fpath_save, format="png")
plt.close()

plt.figure(figsize=(9, 4), dpi=200)
sns.histplot(data=stepsize, stat="count", bins=50, color="indigo", alpha=0.5)
plt.title("Stepsize Distribution")
plt.xlabel("Step Size, nm")
plt.tight_layout()
fpath_save = join(folder_save, "StepsizeDistribution.png")
plt.savefig(fpath_save, format="png")
plt.close()
