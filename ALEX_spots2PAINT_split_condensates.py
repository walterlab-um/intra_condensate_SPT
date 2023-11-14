from tifffile import imwrite
from skimage.util import img_as_uint
from tkinter import filedialog as fd
import os
from os.path import dirname, basename
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
import cv2
import math
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import pandas as pd
from rich.progress import track
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

print(
    "Choose BOTH left-spots_reformatted AND right-spots_reformatted csv files from ALEX SPT-PAINT experiment:"
)
lst_path = list(fd.askopenfilenames())

folder_save = dirname(lst_path[0])
os.chdir(folder_save)
lst_fname_left = [
    basename(f) for f in lst_path if f.endswith("left-spots_reformatted.csv")
]
lst_fname_right = [
    f.split("left")[0] + "right" + f.split("left")[-1] for f in lst_fname_left
]

tracklength_threshold = 10
condensate_area_threshold = 200  # pixels
box_padding = 3  # pixels padding arround each condensate contour
sum_loc_threshold = 30  # PAINT threshold for summed PAINT signal from both channels

um_per_pixel = 0.117
scaling_factor = 1
um_per_pixel_PAINT = um_per_pixel / scaling_factor
xpixels_ONI = 418
ypixels_ONI = 674
xedges = np.arange((xpixels_ONI + 1) * scaling_factor)
yedges = np.arange((ypixels_ONI + 1) * scaling_factor)


def spots2PAINT(df):
    # This function reconstruct PAINT from the whole dataframe, assuming it covers the full FOV. Therefore, it's not for individual condensates
    # single-frame spots
    df_single_frame_spots = df[df["trackID"].isna()]
    img_spots, _, _ = np.histogram2d(
        x=df_single_frame_spots["x"].to_numpy(float) * scaling_factor,
        y=df_single_frame_spots["y"].to_numpy(float) * scaling_factor,
        bins=(xedges, yedges),
    )

    lst_tracklength = []
    # tracks
    df_tracks = df[df["trackID"].notna()]
    all_trackID = df_tracks["trackID"].unique()
    lst_of_arr_x = []
    lst_of_arr_y = []
    for trackID in track(all_trackID, description="Reconstruction: PAINT"):
        df_current = df_tracks[df_tracks["trackID"] == trackID]
        lst_tracklength.append(df_current.shape[0])
        # for short tracks, treat as spots
        if df_current.shape[0] <= tracklength_threshold:
            lst_of_arr_x.append(df_current["x"].to_numpy(float) * scaling_factor)
            lst_of_arr_y.append(df_current["y"].to_numpy(float) * scaling_factor)
            continue
        # for long tracks, randomly pick tracklength_threshold number of spots
        else:
            chosen_idx = np.random.choice(df_current.shape[0], tracklength_threshold)
            lst_of_arr_x.append(
                df_current.iloc[chosen_idx]["x"].to_numpy(float) * scaling_factor
            )
            lst_of_arr_y.append(
                df_current.iloc[chosen_idx]["y"].to_numpy(float) * scaling_factor
            )
            continue

    img_tracks, _, _ = np.histogram2d(
        x=np.hstack(lst_of_arr_x),
        y=np.hstack(lst_of_arr_y),
        bins=(xedges, yedges),
    )

    img_PAINT = img_spots + img_tracks

    return img_PAINT


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

    if box_x_range[1] > xpixels_ONI:
        box_x_max = xpixels_ONI
    else:
        box_x_max = box_x_range[1]

    if box_y_range[0] < 0:
        box_y_min = 0
    else:
        box_y_min = box_y_range[0]

    if box_y_range[1] > ypixels_ONI:
        box_y_max = ypixels_ONI
    else:
        box_y_max = box_y_range[1]

    box_x_range_final = (box_x_min, box_x_max)
    box_y_range_final = (box_y_min, box_y_max)

    return box_x_range_final, box_y_range_final


def plt_cnt_tracks_individual(
    img, cnt, df_track, vmin, vmax, box_x_range, box_y_range, cmap, fpath
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
    all_trackID = df_track["trackID"].unique()
    for trackID in all_trackID:
        plt.plot(
            df_track[df_track["trackID"] == trackID].y,
            df_track[df_track["trackID"] == trackID].x,
            ".-k",
            alpha=0.05,
        )
    plt.savefig(fpath, dpi=300, format="png", bbox_inches="tight")
    plt.close()


def center_track_coordinates(df_in, box_x_range, box_y_range):
    selector = (
        (df_in.x > box_x_range[0])
        & (df_in.x < box_x_range[1])
        & (df_in.y > box_y_range[0])
        & (df_in.y < box_y_range[1])
    )
    df_in_box = df_in.loc[selector]
    # -0.5 to counter the fact that imshow grid starts from the edge not center
    track_x = df_in_box["x"].to_numpy(float) - box_x_range[0] - 0.5
    track_y = df_in_box["y"].to_numpy(float) - box_y_range[0] - 0.5
    df_in_box["x"] = track_x
    df_in_box["y"] = track_y
    return df_in_box


def single_condensate_stepsize_img(df_track, img_shape):
    ## Reconstruct step size iamge, unit: um
    lst_mid_x = []
    lst_mid_y = []
    lst_stepsize = []
    all_trackID = df_track["trackID"].unique()
    for trackID in all_trackID:
        df_current = df_track[df_track["trackID"] == trackID]
        xs = df_current["x"].to_numpy(float)
        ys = df_current["y"].to_numpy(float)
        mid_xs = (xs[1:] + xs[:-1]) / 2
        mid_ys = (ys[1:] + ys[:-1]) / 2
        steps = (
            np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2) * um_per_pixel
        )
        lst_mid_x.extend(mid_xs)
        lst_mid_y.extend(mid_ys)
        lst_stepsize.extend(steps)

    df_all_steps = pd.DataFrame(
        {
            "mid_x": lst_mid_x,
            "mid_y": lst_mid_y,
            "stepsize": lst_stepsize,
        },
        dtype=float,
    )

    # put them in grid, calculate mean
    img_stepsize = np.zeros(img_shape)
    for x in range(img_stepsize.shape[0]):
        for y in range(img_stepsize.shape[1]):
            df_current = df_all_steps[
                df_all_steps["mid_x"].between(x, x + 1)
                & df_all_steps["mid_y"].between(y, y + 1)
            ]
            mean_stepsize = df_current["stepsize"].mean()
            img_stepsize[x, y] = mean_stepsize

    return img_stepsize


def smooth_stepsize_img(img_stepsize, sigma):
    img_stepsize_no_nan = np.nan_to_num(img_stepsize)
    img_stepsize_smoothed = gaussian_filter(img_stepsize_no_nan, sigma)
    return img_stepsize_smoothed


def cnt2mask(imgshape, cnt):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    cv2.fillPoly(mask, [cnt], (255))
    mask = mask != 0

    return mask


def plot_correlation(x, y, xlabel, ylabel, fname_save):
    plt.figure(figsize=(3, 3))
    plt.scatter(x, y, color="gray")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    rho, _ = pearsonr(x, y)
    plt.title(r"$\rho$ = " + str(round(rho, 2)), fontsize=11)
    plt.savefig(fname_save, dpi=300, format="png", bbox_inches="tight")
    plt.close()


for fname_left, fname_right in zip(lst_fname_left, lst_fname_right):
    print("Now processing:", fname_left.split("-left")[0])
    ## Reconstruct PAINT image
    df_left = pd.read_csv(fname_left)
    img_PAINT_left = spots2PAINT(df_left)
    print("Left channel PAINT reconstruction done.")
    df_right = pd.read_csv(fname_right)
    img_PAINT_right = spots2PAINT(df_right)
    print("Right channel PAINT reconstruction done.")

    ## Split to individual condensates
    img_denoise = gaussian_filter(img_PAINT_left + img_PAINT_right, sigma=1)
    edges = img_denoise > 10
    # find contours coordinates in binary edge image. contours here is a list of np.arrays containing all coordinates of each individual edge/contour.
    contours, _ = cv2.findContours(edges * 1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # filter out small condensates
    contours_filtered = [
        cnt for cnt in contours if cv2.contourArea(cnt) > condensate_area_threshold
    ]

    condensateID = 0
    for cnt in track(
        contours_filtered, description="Reconstructing individual condensates"
    ):
        ## determine a square box for individual condensate
        box_x_range, box_y_range = cnt2box(cnt)

        ## center both cnt and tracks coordinates to the center of the box
        cnt_centered = deepcopy(cnt)
        cnt_centered[:, 0][:, 0] = cnt_centered[:, 0][:, 0] - box_y_range[0]
        cnt_centered[:, 0][:, 1] = cnt_centered[:, 0][:, 1] - box_x_range[0]
        df_left_inbox = center_track_coordinates(df_left, box_x_range, box_y_range)
        df_right_inbox = center_track_coordinates(df_right, box_x_range, box_y_range)

        ## crop img_PAINT by the box
        img_PAINT_left_inbox = img_PAINT_left[
            box_x_range[0] : box_x_range[1], box_y_range[0] : box_y_range[1]
        ]
        img_PAINT_right_inbox = img_PAINT_right[
            box_x_range[0] : box_x_range[1], box_y_range[0] : box_y_range[1]
        ]

        ## Calculate step size image
        box_shape = (np.ptp(box_x_range), np.ptp(box_y_range))
        img_stepsize_left = single_condensate_stepsize_img(df_left_inbox, box_shape)
        img_stepsize_right = single_condensate_stepsize_img(df_right_inbox, box_shape)
        img_stepsize_left_smoothed = smooth_stepsize_img(img_stepsize_left, 1)
        img_stepsize_right_smoothed = smooth_stepsize_img(img_stepsize_right, 1)

        ## save csv within box, img_PAINT, img_stepsize, and a plot with img_PAINT+cnt+tracks
        fname_save_prefix = (
            fname_left.split("-left")[0]
            + "-threshold-"
            + str(tracklength_threshold)
            + "-condensateID-"
            + str(condensateID)
            + "-"
        )
        df_left_inbox.to_csv(fname_save_prefix + "left.csv", index=False)
        df_right_inbox.to_csv(fname_save_prefix + "right.csv", index=False)
        imwrite(fname_save_prefix + "left-PAINT.tif", img_PAINT_left_inbox)
        imwrite(fname_save_prefix + "right-PAINT.tif", img_PAINT_right_inbox)
        imwrite(fname_save_prefix + "left-stepsize.tif", img_stepsize_left_smoothed)
        imwrite(fname_save_prefix + "right-stepsize.tif", img_stepsize_right_smoothed)
        plt_cnt_tracks_individual(
            img_PAINT_left_inbox,
            cnt,
            df_left_inbox,
            0,
            20,
            box_x_range,
            box_y_range,
            "Blues",
            fname_save_prefix + "left-PAINT_cnt_tracks.png",
        )
        plt_cnt_tracks_individual(
            img_PAINT_right_inbox,
            cnt,
            df_right_inbox,
            0,
            20,
            box_x_range,
            box_y_range,
            "Reds",
            fname_save_prefix + "right-PAINT_cnt_tracks.png",
        )

        ## Plot correlations
        mask = img_PAINT_left_inbox + img_PAINT_right_inbox > sum_loc_threshold
        plot_correlation(
            img_PAINT_left_inbox[mask],
            img_PAINT_right_inbox[mask],
            "FUS PAINT, #/pixel",
            "RNA PAINT, #/pixel",
            fname_save_prefix + "FlocRloc.png",
        )
        plot_correlation(
            img_stepsize_left_smoothed[mask],
            img_stepsize_right_smoothed[mask],
            r"FUS Step Size, $\mu$m",
            r"RNA Step Size, $\mu$m",
            fname_save_prefix + "FstepRstep.png",
        )
        plot_correlation(
            img_PAINT_left_inbox[mask],
            img_stepsize_right_smoothed[mask],
            "FUS PAINT, #/pixel",
            r"RNA Step Size, $\mu$m",
            fname_save_prefix + "FlocRstep.png",
        )
        plot_correlation(
            img_stepsize_left_smoothed[mask],
            img_PAINT_right_inbox[mask],
            r"FUS Step Size, $\mu$m",
            "RNA PAINT, #/pixel",
            fname_save_prefix + "FstepRloc.png",
        )
        plot_correlation(
            img_PAINT_left_inbox[mask],
            img_stepsize_left_smoothed[mask],
            "FUS PAINT, #/pixel",
            r"FUS Step Size, $\mu$m",
            fname_save_prefix + "FlocFstep.png",
        )
        plot_correlation(
            img_PAINT_right_inbox[mask],
            img_stepsize_right_smoothed[mask],
            "RNA PAINT, #/pixel",
            r"RNA Step Size, $\mu$m",
            fname_save_prefix + "RlocRstep.png",
        )


# def pltcontours_all(img, contours, vmin, vmax):
#     plt.figure()
#     # Contrast stretching
#     plt.imshow(img, cmap="Blues", vmin=vmin, vmax=vmax)
#     for cnt in contours:
#         x = cnt[:, 0][:, 0]
#         y = cnt[:, 0][:, 1]
#         plt.plot(x, y, "-", color="black", linewidth=2)
#         # still the last closing line will be missing, get it below
#         xlast = [x[-1], x[0]]
#         ylast = [y[-1], y[0]]
#         plt.plot(xlast, ylast, "-", color="black", linewidth=2)
#     plt.xlim(0, img.shape[0])
#     plt.ylim(0, img.shape[1])
#     plt.tight_layout()
#     plt.axis("scaled")
#     plt.axis("off")
