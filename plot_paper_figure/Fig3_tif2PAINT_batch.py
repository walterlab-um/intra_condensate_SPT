from tifffile import imread
import os
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'


folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_coralled by nano domains/FUS488_FL_PAINT_final"
os.chdir(folder_save)
lst_files = [f for f in os.listdir(".") if f.endswith(".tif")]

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


def plot_scalebar():
    scale_bar_offset = 2
    scalebar_length_um = 1
    scalebar_length_pxl = scalebar_length_um / 0.117
    plt.plot(
        [scale_bar_offset, scale_bar_offset + scalebar_length_pxl],
        [scale_bar_offset, scale_bar_offset],
        "-",
        color="black",
        lw=5,
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
    plot_scalebar()
    plt.gca().invert_yaxis()
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
    # plot_scalebar()
    plt.gca().invert_yaxis()
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
    # plot_scalebar()
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(
        fname[:-4] + "-overlay.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close()

    # Correlation
    plt.figure(figsize=(3, 3))
    plt.scatter(
        img_red.flat,
        img_blue.flat,
        color="gray",
    )
    # plt.axis("equal")
    plt.xlim(0, img_red.max())
    plt.ylim(0, img_blue.max())
    plt.xlabel(
        "RNA, # per pxl",
        # weight="bold",
        fontsize=20,
    )
    plt.ylabel(
        "FUS, # per pxl",
        # weight="bold",
        fontsize=20,
    )
    rho, pval = pearsonr(img_red.flat, img_blue.flat)
    plt.title(
        r"$\rho$ = " + str(round(rho, 2)),
        weight="bold",
        fontsize=30,
    )
    plt.savefig(
        fname[:-4] + "-pearson.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
