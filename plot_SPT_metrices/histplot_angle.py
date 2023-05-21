import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(color_codes=True, style="white")

folder_path = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup/Verify_alpha-angledist_on_mobile_vs_constrained"
os.chdir(folder_path)

lst_fname = [f for f in os.listdir(folder_path) if f.startswith("Angles_")]

for fname in lst_fname:
    df_angles = pd.read_csv(fname)

    # Per Track
    bins = np.linspace(0, 180, df_angles.shape[1] - 3).astype(int)

    hist_per_track = (
        df_angles[df_angles["N_steps"] > 100].dropna().to_numpy()[:, 4:].astype(float)
    )
    hist_per_track_mean = np.nanmean(hist_per_track, axis=0)
    hist_per_track_std = np.nanstd(hist_per_track, axis=0)

    plt.figure(figsize=(5, 4), dpi=300)
    plt.errorbar(
        x=bins[:-1] + (bins[1] - bins[0]) / 2,
        y=hist_per_track_mean,
        yerr=hist_per_track_std,
        ls="-",
        color="#9a3324",
        lw=2,
        capsize=5,
        capthick=3,
    )
    plt.title(fname[7:-4] + " Per Track", weight="bold")
    plt.xlabel("Angle between Two Steps, Degree", weight="bold")
    plt.ylabel("Probability", weight="bold")
    plt.xlim(0, 180)
    plt.tight_layout()
    plt.savefig("AngleHist_PerTrack_" + fname[7:-4] + ".png", format="png")
    plt.close()
