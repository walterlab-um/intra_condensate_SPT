import os
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(color_codes=True, style="white")

###############################################
# Loading Data
folderpath = (
    "/Volumes/AnalysisGG/RAW_DATA_ORGANIZED/20220712_10FUS_1Mg_10Dex_noTotR_24C/"
)
subfolder_low = "tracks_lowfreq"
subfolder_high = "tracks_highfreq"
R2_threshold = 0.7


lst_files_low = [
    f for f in os.listdir(join(folderpath, subfolder_low)) if f.endswith("D.csv")
]
lst_files_high = [
    f for f in os.listdir(join(folderpath, subfolder_high)) if f.endswith("D.csv")
]

os.chdir(join(folderpath, subfolder_low))
data_low = np.array([], dtype=float)
for file in lst_files_low:
    df = pd.read_csv(file)
    df = df[df["R2"] > R2_threshold]
    data_low = np.append(data_low, df["log10D"].to_numpy(dtype=float))
os.chdir(join(folderpath, subfolder_high))
data_high = np.array([], dtype=float)
for file in lst_files_high:
    df = pd.read_csv(file)
    df = df[df["R2"] > R2_threshold]
    data_high = np.append(data_high, df["log10D"].to_numpy(dtype=float))


plt.figure(figsize=(9, 4), dpi=200)
sns.histplot(
    data=data_low,
    binwidth=0.1,
    stat="probability",
    color=sns.color_palette()[0],
    label="0.5 Hz Sampling",
    alpha=0.6,
)
sns.histplot(
    data=data_high,
    binwidth=0.1,
    stat="probability",
    color=sns.color_palette()[3],
    label="50 Hz Sampling",
    alpha=0.6,
)
plt.legend()
# plt.ylim(0, 1)
plt.xlabel("log10(D), \u03BCm$^{2}$/s", weight="bold", fontsize=15)
plt.ylabel("Frequency", weight="bold", fontsize=15)
plt.tight_layout()
os.chdir(folderpath)
plt.savefig("log10D histogram.png", format="png")
