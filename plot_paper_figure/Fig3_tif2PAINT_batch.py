from tifffile import imread
import os
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import pandas as pd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'


folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_coralled by nano domains/FUS488_miR21_PAINT_final/not_chosen_ones"
os.chdir(folder_save)
lst_files = [f for f in os.listdir(".") if f.endswith(".tif")]

line_color = "white"  # #00274C
scalebar_color = "white"

colors = [(0, 0, 1, c) for c in np.linspace(0, 1, 100)]
cmap_name = "transparant2blue"
cmap_blue = clr.LinearSegmentedColormap.from_list(
    cmap_name,
    colors,
    N=100,
)

colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
cmap_name = "transparant2red"
cmap_red = clr.LinearSegmentedColormap.from_list(
    cmap_name,
    colors,
    N=100,
)

for fname in track(lst_files):
    img = imread(fname)
    img_red = img[0, :, :]
    img_blue = img[1, :, :]

    plt.figure()
    vmin, vmax = np.quantile(
        img_blue.reshape(-1),
        (0.1, 0.95),
    )
    plt.imshow(
        img_blue,
        cmap=cmap_blue,
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis("off")
    plt.savefig(
        fname[:-4] + "-blue.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    vmin, vmax = np.quantile(
        img_red.reshape(-1),
        (0.1, 0.99),
    )
    plt.imshow(
        img_red,
        cmap=cmap_red,
        vmin=vmin,
        vmax=vmax,
    )
    plt.axis("off")
    plt.savefig(
        fname[:-4] + "-red.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    plt.figure()
    vmin, vmax = np.quantile(
        img_blue.reshape(-1),
        (0.1, 0.95),
    )
    plt.imshow(
        img_blue,
        cmap=cmap_blue,
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
    )
    vmin, vmax = np.quantile(
        img_red.reshape(-1),
        (0.1, 0.99),
    )
    plt.imshow(
        img_red,
        cmap=cmap_red,
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
    )
    plt.axis("off")
    plt.savefig(
        fname[:-4] + "-overlay.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close()
