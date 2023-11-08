from tifffile import imread
import os
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from matplotlib import cm
import numpy as np
import pandas as pd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'


line_color = "white"  # #00274C
scalebar_color = "white"

folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_coralled by nano domains/FUS488_FLIM/20230918_chosen_ones"
os.chdir(folder_save)

lst_files_tau = [f for f in os.listdir(".") if f.endswith("tau_1 raw.tif")]
lst_files_chi = [f for f in os.listdir(".") if f.endswith("chi2 raw.tif")]

cmap = mpl.colormaps.get_cmap("PuOr")
cmap.set_bad(color="black")

# tau plots
for fname in track(lst_files_tau):
    img = imread(fname)
    img = img / 1e3
    img[img == 0] = np.nan
    plt.figure()
    vmin, vmax = np.nanquantile(
        img.reshape(-1),
        [
            0.05,
            0.95,
        ],
    )
    plt.imshow(
        img,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    # plot color bar for time
    cbar = plt.colorbar(
        cm.ScalarMappable(norm=clr.Normalize(vmin, vmax), cmap=cmap),
        ax=plt.gca(),
        orientation="vertical",
        pad=0.05,
        drawedges=False,
    )
    cbar.set_label(
        label="Life Time, ns",
        fontsize=21,
        labelpad=-10,
        family="Arial",
    )
    cbar.set_ticks(
        [
            round(vmin, 2),
            round(vmax, 2),
        ]
    )
    cbar.ax.tick_params(labelsize=21)

    plt.gca().set_facecolor("xkcd:salmon")
    plt.axis("off")
    plt.savefig(
        fname[:-4] + ".png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close()
