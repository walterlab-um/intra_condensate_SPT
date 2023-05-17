import os
import matplotlib.pyplot as plt
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.colors as colors

sns.set(color_codes=True, style="white")

print("Select a MSD-tau-alltracks.csv file to be plot")
fpath = fd.askopenfilename()
print("Enter the time between frames (seconds):")
# s_per_frame = float(input())
s_per_frame = 0.02
# fpath = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/bioFUStether-10FUS-1Mg-10Dex-RT/NoTotalRNA/FL-0min/tracks-rawRNA-minlength-5/MSD-tau-alltracks.csv"
print("Type in a title below:")
title = input()

data = pd.read_csv(fpath)

all_MSD = np.array(data.MSD_um2, dtype=float)
all_tau = np.array(data.tau, dtype=int)
track_start_selector = np.concatenate(([True], list(all_tau[1:] - all_tau[:-1] < 1)))
track_end_selector = np.append(all_tau[1:] - all_tau[:-1] < 1, True)
track_start_index = np.where(track_start_selector)[0]
track_end_index = np.where(track_end_selector)[0]
fig, ax = plt.subplots(dpi=600)
for start, end in zip(track_start_index, track_end_index):
    singletrack_MSD = all_MSD[start : end + 1]
    singletrack_tau = all_tau[start : end + 1] * s_per_frame * 1e3  # unit: ms
    ax.plot(singletrack_tau, singletrack_MSD, "-", color="gray", alpha=0.1)
# plt.xlim(all_tau.min() * s_per_frame * 1e3, all_tau.max() * s_per_frame * 1e3)
# plt.ylim(all_MSD.min(), all_MSD.max())
plt.xlim(0, 300)
plt.ylim(0, 5)
plt.title(title, weight="bold")
plt.xlabel("tau, ms", weight="bold")
plt.ylabel("MSD, $\mu$m$^2$", weight="bold")
plt.tight_layout()
fpath_save = fpath.strip(".csv") + "-overlay.png"
plt.savefig(fpath_save, format="png")
