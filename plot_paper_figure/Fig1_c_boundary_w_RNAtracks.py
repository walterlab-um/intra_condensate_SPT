import os
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import pandas as pd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

# full size: 418x674
zoom_in_x = (90, 200)
zoom_in_y = (280, 390)

um_per_pixel = 0.117
folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig1_system design/c_boundary_w_RNAtracks/"
os.chdir(folder_save)

line_color = "#00274C"
scalebar_color = "black"
cmap_RNA = clr.LinearSegmentedColormap.from_list(
    "black2red", ["#333232", "#9a3324"], N=200
)

# scale bar
scalebar_length_um = 1
um_per_pixel = 0.117
scalebar_length_pxl = scalebar_length_um / um_per_pixel

# load dataset
fname_RNA = "SPT_results_AIO-20221031-FL_noTR_noDex_20ms_0hr_Replicate1_FOV-7-RNAs.csv"
fname_condensate = "condensates_AIO-20221031-FL_noTR_noDex_20ms_0hr_Replicate1_FOV-7-condensates_AveProj_Simple Segmentation.csv"

df_RNA = pd.read_csv(fname_RNA)
df_condensate = pd.read_csv(fname_condensate)


def list_like_string_to_xyt(list_like_string):
    # example list_like_string structure of xyt: '[0, 1, 2, 3]'
    list_of_xyt_string = list_like_string[1:-1].split(", ")
    lst_xyt = []
    for xyt_string in list_of_xyt_string:
        lst_xyt.append(float(xyt_string))

    return np.array(lst_xyt, dtype=float)


plt.figure(dpi=300)
## plot condensate boundaries
for condensateID in track(
    df_condensate["condensateID"].unique(), description="Plot condensates"
):
    str_condensate_coords = df_condensate[
        df_condensate["condensateID"] == condensateID
    ]["contour_coord"].squeeze()
    x = []
    y = []
    for str_condensate_xy in str_condensate_coords[2:-2].split("], ["):
        xy = str_condensate_xy.split(", ")
        x.append(int(xy[0]))
        y.append(int(xy[1]))
    plt.plot(x, y, "-", color=line_color, linewidth=2, alpha=0.7)
    # still the last closing line will be missing, get it below
    xlast = [x[-1], x[0]]
    ylast = [y[-1], y[0]]
    plt.plot(xlast, ylast, "-", color=line_color, linewidth=2, alpha=0.7)
# plt.xlim(0, 418)
# plt.ylim(0, 674)

## plot RNA tracks
for _, row in track(df_RNA.iterrows(), description="Plot RNAs"):
    x = list_like_string_to_xyt(row["list_of_x"])
    y = list_like_string_to_xyt(row["list_of_y"])
    t = list_like_string_to_xyt(row["list_of_t"])

    for i in range(len(t) - 1):
        if (
            (x[i] > zoom_in_x[0])
            & (x[i] < zoom_in_x[1])
            & (y[i] > zoom_in_y[0])
            & (y[i] < zoom_in_y[1])
        ):
            plt.plot(
                x[i : i + 2],
                y[i : i + 2],
                "-",
                color=cmap_RNA(np.mean(t[i : i + 2]) / 200),
                linewidth=0.5,
            )

plt.tight_layout()
plt.axis("scaled")
plt.axis("off")
## plot scale bar

# non-zoom in
# plt.plot([10, 10 + scalebar_length_pxl], [10, 10], "-", color=scalebar_color, lw=5)

# zoom in
plt.xlim(zoom_in_x[0], zoom_in_x[1])
plt.ylim(zoom_in_y[0], zoom_in_y[1])
plt.plot(
    [zoom_in_x[0] + 5, zoom_in_x[0] + 5 + scalebar_length_pxl],
    [zoom_in_y[0] + 5, zoom_in_y[0] + 5],
    "-",
    color=scalebar_color,
    lw=5,
)
# plt.show()
plt.savefig("Fig1_c_boundary_w_RNAtracks_zoomin.png", format="png", bbox_inches="tight")
plt.close()
