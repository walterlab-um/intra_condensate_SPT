import os
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import numpy as np
import pandas as pd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

um_per_pixel = 0.117
folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig1_system design/c_boundary_w_RNAtracks/"
os.chdir(folder_save)

line_color = "#00274C"
scalebar_color = "black"
cmap_RNA = clr.LinearSegmentedColormap.from_list(
    "black2red", ["#333232", "#9a3324"], N=200
)


# scale bar
scalebar_length_um = 5
um_per_pixel = 0.117
scalebar_length_pxl = scalebar_length_um / um_per_pixel

# loading datasets
keyword = "Replicate1_FOV-6-"

fpath_RNA = [f for f in os.listdir(".") if f.startswith("SPT_results")][0]
fpath_condensates = [f for f in os.listdir(".") if f.startswith("condensates")][0]

df_RNA = pd.read_csv(fpath_RNA)
df_condensate = pd.read_csv(fpath_condensates)

df_RNA_current_FOV = df_RNA[df_RNA["filename"].str.contains(keyword)]
# load condensates near the RNA as dictionary of polygons
df_condensate_current_FOV = df_condensate[
    df_condensate["filename"].str.contains(keyword)
]


def list_like_string_to_xyt(list_like_string):
    # example list_like_string structure of xyt: '[0, 1, 2, 3]'
    list_of_xyt_string = list_like_string[1:-1].split(", ")
    lst_xyt = []
    for xyt_string in list_of_xyt_string:
        lst_xyt.append(float(xyt_string))

    return np.array(lst_xyt, dtype=float)


plt.figure(dpi=1000)
## plot condensate boundaries
for condensateID in track(
    df_condensate_current_FOV["condensateID"].unique(), description="Plot condensates"
):
    str_condensate_coords = df_condensate_current_FOV[
        df_condensate_current_FOV["condensateID"] == condensateID
    ]["contour_coord"].squeeze()
    x = []
    y = []
    for str_condensate_xy in str_condensate_coords[2:-2].split("], ["):
        xy = str_condensate_xy.split(", ")
        x.append(int(xy[0]))
        y.append(int(xy[1]))
    plt.plot(x, y, "-", color=line_color, linewidth=0.5)
    # still the last closing line will be missing, get it below
    xlast = [x[-1], x[0]]
    ylast = [y[-1], y[0]]
    plt.plot(xlast, ylast, "-", color=line_color, linewidth=0.5)
plt.xlim(0, 418)
plt.ylim(0, 674)

## plot RNA tracks
for _, row in track(df_RNA_current_FOV.iterrows(), description="Plot RNAs"):
    x = list_like_string_to_xyt(row["list_of_x"])
    y = list_like_string_to_xyt(row["list_of_y"])
    t = list_like_string_to_xyt(row["list_of_t"])

    for i in range(len(t) - 1):
        plt.plot(
            x[i : i + 2],
            y[i : i + 2],
            "-",
            color=cmap_RNA(np.mean(t[i : i + 2]) / 200),
            linewidth=0.1,
        )

## plot scale bar
plt.plot([10, 10 + scalebar_length_pxl], [10, 10], "-", color=scalebar_color, lw=5)
plt.axis("scaled")
plt.axis("off")
plt.savefig("Fig1_c_boundary_w_RNAtracks.png", format="png", bbox_inches="tight")
plt.close()
