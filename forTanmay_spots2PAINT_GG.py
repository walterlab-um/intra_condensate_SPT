from tifffile import imwrite
from tkinter import filedialog as fd
import os
from os.path import dirname, basename, join
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle
from rich.progress import track

matplotlib.use("Agg")
pd.options.mode.chained_assignment = None  # default='warn'

folder_save = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/RAW_DATA_ORGANIZED/TanmayData"
os.chdir(folder_save)
fname = "particles.csv"

condensate_area_threshold = 20  # pixels
box_padding = 3  # pixels padding arround each condensate contour

um_per_pixel = 0.117
scaling_factor = 3
um_per_pixel_PAINT = um_per_pixel / scaling_factor
xpixels_ONI = 418
ypixels_ONI = 674
xedges = np.arange((xpixels_ONI + 1) * scaling_factor)
yedges = np.arange((ypixels_ONI + 1) * scaling_factor)

columns = [
    "condensateID",
    "contour_coord",
    "center_x_pxl",
    "center_y_pxl",
    "area_um2",
    "R_nm",
    "mean_intensity",
    "max_intensity",
    "max_location",
    "aspect_ratio",
    "contour_solidity",
    "contour_extent",
]


def spots2PAINT(df):
    # single-frame spots
    df_single_frame_spots = df
    img_spots, _, _ = np.histogram2d(
        x=df_single_frame_spots["x"].to_numpy(float) * scaling_factor,
        y=df_single_frame_spots["y"].to_numpy(float) * scaling_factor,
        bins=(xedges, yedges),
    )

    return img_spots


def cnt2box(cnt):
    # Note that the x and y of contours coordinates are swapped in cv2
    center_x = (cnt[:, 0][:, 1].max() + cnt[:, 0][:, 1].min()) / 2
    range_x = cnt[:, 0][:, 1].max() - cnt[:, 0][:, 1].min()
    center_y = (cnt[:, 0][:, 0].max() + cnt[:, 0][:, 0].min()) / 2
    range_y = cnt[:, 0][:, 0].max() - cnt[:, 0][:, 0].min()
    box_halfwidth = (np.max([range_x, range_y]) + box_padding * 2) / 2
    box_x_range = (
        math.floor(center_x - box_halfwidth),
        math.ceil(center_x + box_halfwidth),
    )
    box_y_range = (
        math.floor(center_y - box_halfwidth),
        math.ceil(center_y + box_halfwidth),
    )
    if box_x_range[0] < 0:
        box_x_min = 0
    else:
        box_x_min = box_x_range[0]

    if box_x_range[1] > xpixels_ONI * scaling_factor:
        box_x_max = xpixels_ONI * scaling_factor
    else:
        box_x_max = box_x_range[1]

    if box_y_range[0] < 0:
        box_y_min = 0
    else:
        box_y_min = box_y_range[0]

    if box_y_range[1] > ypixels_ONI * scaling_factor:
        box_y_max = ypixels_ONI * scaling_factor
    else:
        box_y_max = box_y_range[1]

    box_x_range_final = (box_x_min, box_x_max)
    box_y_range_final = (box_y_min, box_y_max)

    return box_x_range_final, box_y_range_final


def plt_cnt_PAINT_individual(
    img, cnt, vmin, vmax, box_x_range, box_y_range, cmap, fpath
):
    plt.figure()
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    x = cnt[:, 0][:, 0]
    x = x - box_y_range[0]
    y = cnt[:, 0][:, 1]
    y = y - box_x_range[0]
    plt.plot(x, y, "-", color="black", linewidth=2)
    xlast = [x[-1], x[0]]
    ylast = [y[-1], y[0]]
    plt.plot(xlast, ylast, "-", color="black", linewidth=2)
    plt.savefig(fpath, dpi=150, format="png", bbox_inches="tight")
    plt.close()


def cnt2mask(imgshape, contours):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    for cnt in contours:
        cv2.fillPoly(mask, [cnt], (255))
    return mask


def cnt_fill(imgshape, cnt):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    cv2.fillPoly(mask, [cnt], (255))

    return mask


def cnt_to_list(cnt):
    # convert cv2's cnt format to list of corrdinates, for the ease of saving in one dataframe cell
    cnt_2d = np.reshape(cnt, (cnt.shape[0], cnt.shape[2]))
    lst_cnt = [cnt_2d[i, :].tolist() for i in range(cnt_2d.shape[0])]

    return lst_cnt


## Reconstruct PAINT image
df = pd.read_csv(fname, sep="\t", index_col=False)
df = df.rename(columns={"X (px)": "x", "Y (px)": "y"})
img_PAINT = spots2PAINT(df)
imwrite("PAINT_whole_FOV.tif", img_PAINT)

## Split to individual condensates
img_denoise = gaussian_filter(img_PAINT, sigma=0.5)
edges = img_denoise > 5
# find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
contours, _ = cv2.findContours(edges * 1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Merge overlapping contours
mask = cnt2mask(img_denoise.shape, contours)
contours_final, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# filter out small condensates
contours_filtered = [
    cnt for cnt in contours_final if cv2.contourArea(cnt) > condensate_area_threshold
]

# Plot and save individual images only for large enough vesicles
lst_rows_of_df = []
condensateID = 0
for cnt in track(
    contours_filtered, description="Reconstructing individual condensates"
):
    # ignore contour if it's as large as the FOV, becuase cv2 recognizes the whole image as a contour
    if cv2.contourArea(cnt) > 0.8 * img_PAINT.shape[0] * img_PAINT.shape[1]:
        continue

    # condensate center coordinates
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        continue
    center_x_pxl = int(M["m10"] / M["m00"])
    center_y_pxl = int(M["m01"] / M["m00"])
    # condensate size
    area_um2 = cv2.contourArea(cnt) * um_per_pixel_PAINT**2
    R_nm = np.sqrt(area_um2 / np.pi) * 1000
    # aspect ratio
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    # extent
    rect_area = w * h
    contour_extent = float(cv2.contourArea(cnt)) / rect_area
    # solidity
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    contour_solidity = float(cv2.contourArea(cnt)) / hull_area
    # intensities
    mask_current_condensate = cnt_fill(img_PAINT.shape, cnt)
    mean_intensity = cv2.mean(img_PAINT, mask=mask_current_condensate)[0]
    _, max_intensity, _, max_location = cv2.minMaxLoc(
        img_PAINT, mask=mask_current_condensate
    )

    # Finally, add the new row to the list to form dataframe
    new_row = [
        condensateID,
        cnt_to_list(cnt),
        center_x_pxl,
        center_y_pxl,
        area_um2,
        R_nm,
        mean_intensity,
        max_intensity,
        max_location,
        aspect_ratio,
        contour_solidity,
        contour_extent,
    ]
    lst_rows_of_df.append(new_row)

    ## determine a square box for individual condensate
    box_x_range, box_y_range = cnt2box(cnt)

    ## center both cnt and tracks coordinates to the center of the box
    cnt_centered = deepcopy(cnt)
    cnt_centered[:, 0][:, 0] = cnt_centered[:, 0][:, 0] - box_y_range[0]
    cnt_centered[:, 0][:, 1] = cnt_centered[:, 0][:, 1] - box_x_range[0]
    # df_inbox = center_track_coordinates(df, box_x_range, box_y_range)

    ## crop img_PAINT by the box
    img_PAINT_inbox = img_PAINT[
        box_x_range[0] : box_x_range[1], box_y_range[0] : box_y_range[1]
    ]
    img_denoised_inbox = img_denoise[
        box_x_range[0] : box_x_range[1], box_y_range[0] : box_y_range[1]
    ]

    ## save csv within box, img_PAINT, img_stepsize, and a plot with img_PAINT+cnt+tracks
    fname_save_prefix = (
        "thresholdArea-"
        + str(condensate_area_threshold)
        + "pxl-condensateID-"
        + str(condensateID)
    )
    # df_inbox.to_csv(fname_save_prefix + ".csv", index=False)
    imwrite(fname_save_prefix + "-PAINT.tif", img_PAINT_inbox)
    imwrite(fname_save_prefix + "-PAINT-denoised.tif", img_denoised_inbox)
    # pickle.dump(cnt_centered, open(fname_save_prefix + "cnt_centered.p", "wb"))
    if condensateID % 50 == 0:
        plt_cnt_PAINT_individual(
            img_PAINT_inbox,
            cnt,
            0,
            10,
            box_x_range,
            box_y_range,
            "Reds",
            fname_save_prefix + "-PAINT_cnt_tracks.png",
        )
    condensateID += 1

df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
fname_save = join(folder_save, "condensates_AIO-" + fname[:-4] + ".csv")
df_save.to_csv(fname_save, index=False)
