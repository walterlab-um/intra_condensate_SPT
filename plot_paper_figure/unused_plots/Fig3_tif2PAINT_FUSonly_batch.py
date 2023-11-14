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

cmap_color_start = "black"
cmap_color_end = "#B9DBF4"
cmap_name = "dark2cyan"
cmap_blue = clr.LinearSegmentedColormap.from_list(
    cmap_name,
    [cmap_color_start, cmap_color_end],
    N=200,
)

for fname in track(lst_files):
    img = imread(fname)
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
    # scale bar
    scale_bar_offset = 2
    scalebar_length_um = 1
    scalebar_length_pxl = scalebar_length_um / 0.117
    plt.plot(
        [scale_bar_offset, scale_bar_offset + scalebar_length_pxl],
        [scale_bar_offset, scale_bar_offset],
        "-",
        color=scalebar_color,
        lw=5,
    )
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(
        fname[:-4] + "-FUSonly.png",
        dpi=300,
        format="png",
        bbox_inches="tight",
    )
    plt.close()
