import math
import os
import pickle
from multiprocessing import Pool, cpu_count
from os.path import dirname, join
from tkinter import filedialog as fd

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from tifffile import imread
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'


"""
The program calculates per-location FUS and RNA normalized distance to center distribution of two channel single partical tracking (SPT) data.
"""


def cnt2mask(imgshape, contours):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    for cnt in contours:
        cv2.fillPoly(mask, [cnt], (255))
    return mask


def corr_within_mask(df, mask):
    """
    Take a Polygon mask and a dataframe contain columns 'x' and 'y', and return numpy array of x and y within the mask.
    """
    lst_x = []
    lst_y = []
    for _, row in df.iterrows():
        if Point(row.x, row.y).within(mask):
            lst_x.append(row.x)
            lst_y.append(row.y)
    array_x = np.array(lst_x, dtype=float)
    array_y = np.array(lst_y, dtype=float)
    return array_x, array_y


def filter_perLoc(df):
    scaling_factor = 1
    tracklength_threshold = 10
    # single-frame spots
    df_single_frame_spots = df[df["trackID"].isna()]
    spots_x = df_single_frame_spots.x.to_numpy(float)
    spots_y = df_single_frame_spots.y.to_numpy(float)
    # tracks
    df_tracks = df[df["trackID"].notna()]
    all_trackID = df_tracks["trackID"].unique()
    lst_of_arr_x = []
    lst_of_arr_y = []
    for trackID in all_trackID:
        df_current = df_tracks[df_tracks["trackID"] == trackID]
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

    tracks_x = np.hstack(lst_of_arr_x)
    tracks_y = np.hstack(lst_of_arr_y)

    df_out = pd.DataFrame(
        {
            "x": np.concatenate(
                [
                    spots_x,
                    tracks_x,
                ]
            ),
            "y": np.concatenate(
                [
                    spots_y,
                    tracks_y,
                ]
            ),
        },
        dtype=float,
    )

    return df_out


def filter_perTrack(df):
    scaling_factor = 1
    tracklength_threshold = 3
    df_tracks = df[df["trackID"].notna()]
    all_trackID = df_tracks["trackID"].unique()
    lst_x = []
    lst_y = []
    for trackID in all_trackID:
        df_current = df_tracks[df_tracks["trackID"] == trackID]
        if df_current.shape[0] >= tracklength_threshold:
            lst_x.append(df_current["x"].mean() * scaling_factor)
            lst_y.append(df_current["y"].mean() * scaling_factor)

    tracks_x = np.array(lst_x, dtype=float)
    tracks_y = np.array(lst_y, dtype=float)

    df_out = pd.DataFrame({"x": tracks_x, "y": tracks_y}, dtype=float)

    return df_out


# Function to process a single file
def process_file(
    i,
    lst_fname_left_csv,
    lst_fname_right_csv,
    lst_fname_left_PAINT,
    lst_fname_right_PAINT,
):
    df_left = pd.read_csv(lst_fname_left_csv[i])
    df_right = pd.read_csv(lst_fname_right_csv[i])
    img_PAINT_left = imread(lst_fname_left_PAINT[i])
    img_PAINT_right = imread(lst_fname_right_PAINT[i])

    img = img_PAINT_left + img_PAINT_right
    edges = img >= 1
    contours, _ = cv2.findContours(edges * 1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    mask = cnt2mask(edges.shape, contours)
    contours_final, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # get only the major condensate
    cnt_condensate = contours_final[0]
    for cnt in contours_final:
        if cv2.contourArea(cnt) > cv2.contourArea(cnt_condensate):
            cnt_condensate = cnt

    mask = Polygon(np.squeeze(cnt_condensate))
    center_xy = (mask.centroid.x, mask.centroid.y)
    mask_r = math.sqrt(mask.area / math.pi)

    df_left = filter_perTrack(df_left)
    d2center = np.sqrt(
        (df_left["x"] - center_xy[0]) ** 2 + (df_left["y"] - center_xy[1]) ** 2
    )
    condensate_r = np.repeat(mask_r, df_left.shape[0])
    d2center_norm = d2center / condensate_r
    df_left["d2center"] = d2center
    df_left["condensate_r"] = condensate_r
    df_left["d2center_norm"] = d2center_norm

    df_right = filter_perTrack(df_right)
    d2center = np.sqrt(
        (df_right["x"] - center_xy[0]) ** 2 + (df_right["y"] - center_xy[1]) ** 2
    )
    condensate_r = np.repeat(mask_r, df_right.shape[0])
    d2center_norm = d2center / condensate_r
    df_right["d2center"] = d2center
    df_right["condensate_r"] = condensate_r
    df_right["d2center_norm"] = d2center_norm

    return (
        df_left.shape[0],
        df_right.shape[0],
        df_left,  # save all FUS tracks (x, y, r, d2center, condensate_r, d2center_norm)
        df_right,  # save all FUS tracks (x, y, r, d2center, condensate_r, d2center_norm)
    )


def main():
    print(
        "Choose all tif and csv files to be batch proccessed. The prefix should be the same for (1) left.csv (2) right.csv (3) left-PAINT.tif (4) right-PAINT.tif"
    )
    lst_files = list(fd.askopenfilenames())
    folder_data = dirname(lst_files[0])
    os.chdir(folder_data)
    folder_save = "../"
    fname_save = "distance2center-DataDict-pooled-perTrack.p"

    all_files = os.listdir(".")
    lst_fname_left_csv = [f for f in all_files if f.endswith("left.csv")]
    lst_fname_left_PAINT = [
        f.split("left.csv")[0] + "left-PAINT.tif" for f in lst_fname_left_csv
    ]
    lst_fname_right_csv = [
        f.split("left.csv")[0] + "right.csv" for f in lst_fname_left_csv
    ]
    lst_fname_right_PAINT = [
        f.split("left.csv")[0] + "right-PAINT.tif" for f in lst_fname_left_csv
    ]
    # The tqdm library provides an easy way to visualize the progress of loops.
    pbar = tqdm(total=len(lst_fname_left_csv))

    # Update function for the progress bar
    def update(*a):
        pbar.update()

    # Create a process pool and map the function to the files
    with Pool(cpu_count()) as p:
        results = []
        for i in range(len(lst_fname_left_csv)):
            result = p.apply_async(
                process_file,
                args=(
                    i,
                    lst_fname_left_csv,
                    lst_fname_right_csv,
                    lst_fname_left_PAINT,
                    lst_fname_right_PAINT,
                ),
                callback=update,
            )
            results.append(result)

        # Wait for all processes to finish and gather results
        results = [r.get() for r in results]
    pbar.close()

    # Unpack results into separate lists
    (
        lst_size_FUS,
        lst_size_RNA,
        lst_df_left,
        lst_df_right,
    ) = map(list, zip(*results))

    dict_to_save = {
        "filenames_FUS": lst_fname_left_csv,
        "filenames_RNA": lst_fname_right_csv,
        "lst_N_locations_FUS": lst_size_FUS,
        "lst_N_locations_RNA": lst_size_RNA,
        "lst_df_FUS_d2center": lst_df_left,  # save all FUS tracks (x, y, r, d2center, condensate_r, d2center_norm)
        "lst_df_RNA_d2center": lst_df_right,  # save all FUS tracks (x, y, r, d2center, condensate_r, d2center_norm)
    }
    pickle.dump(
        dict_to_save,
        open(join(folder_save, fname_save), "wb"),
    )
    print("Saved successfully at the following path:")
    print(join(folder_save, fname_save))


if __name__ == "__main__":
    main()