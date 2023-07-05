from tifffile import imread
import os
from os.path import dirname
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

um_per_pixel = 0.117
fpath = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig1_system design/b_condensate_FOV/20221031-FL_noTR_noDex_20ms_Replicate1_FOV-8-condensates_AveProj_Simple Segmentation.tif"
folder_save = dirname(fpath)
os.chdir(folder_save)

plow = 0.5  # imshow intensity percentile
phigh = 85
line_color = "#00274C"
scalebar_color = "#FFA000"

# scale bar
scalebar_length_um = 5
um_per_pixel = 0.117
scalebar_length_pxl = scalebar_length_um / um_per_pixel

ilastik_output = imread(fpath)
img = imread(fpath[:-24] + ".tif")

# Cropping
ilastik_output = ilastik_output[35:235, 150:350]
img = img[35:235, 150:350]

mask_all_condensates = 2 - ilastik_output  # background label=2, condensate label=1
# find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
contours, _ = cv2.findContours(
    mask_all_condensates, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
)


plt.figure()
# Contrast stretching
vmin, vmax = np.percentile(img, (plow, phigh))
plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
for cnt in contours:
    x = cnt[:, 0][:, 0]
    y = cnt[:, 0][:, 1]
    plt.plot(x, y, "-", color=line_color, linewidth=2)
    # still the last closing line will be missing, get it below
    xlast = [x[-1], x[0]]
    ylast = [y[-1], y[0]]
    plt.plot(xlast, ylast, "-", color=line_color, linewidth=2)
plt.xlim(0, img.shape[0])
plt.ylim(0, img.shape[1])
plt.plot([10, 10 + scalebar_length_pxl], [10, 10], "-", color=scalebar_color, lw=7)
plt.axis("scaled")
plt.axis("off")
plt.savefig("Fig1_b_condensate_FOV.png", format="png", bbox_inches="tight", dpi=600)
plt.close()
