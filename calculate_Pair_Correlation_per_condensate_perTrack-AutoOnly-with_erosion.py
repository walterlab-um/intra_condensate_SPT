from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter
from tifffile import imread
import cv2
import math
import os
from os.path import join, dirname
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tkinter import filedialog as fd

pd.options.mode.chained_assignment = None  # default='warn'

"""
The program calculates per-track Pair Correlation Function (PCF; aka. Radial Distribution Function, RDF) of single channel single partical tracking (SPT) data.

Per-track means every single trajectory in the SPT dataset is treated as a single point, using the mean location of the track. Such point set is then subjected to PCF calculation.

It saves four PCF with corresponding informatioon in a single pickle file: auto PCF
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


def erode_polygon(polygon, erosion_distance=5):
    """
    Erodes the given shapely Polygon by a specified distance.

    Parameters:
    - polygon: shapely.geometry.Polygon
        The polygon to be eroded.
    - erosion_distance: float
        The distance by which to erode the polygon. Default is 5.

    Returns:
    - eroded_polygon: shapely.geometry.Polygon
        The eroded polygon.
    """
    # Erode the polygon using a negative buffer
    eroded_polygon = polygon.buffer(-erosion_distance)

    # Ensure the result is still a valid Polygon
    if not eroded_polygon.is_valid:
        raise ValueError("The erosion operation resulted in an invalid polygon.")

    return eroded_polygon


def PairCorr_with_edge_correction(
    df_ref,
    mask,
    nm_per_pxl,
    r_max_nm,
    ringwidth_nm,
    dr_slidingrings_nm,
):
    # only count particles within mask
    x_ref, y_ref = corr_within_mask(df_ref, mask)

    # Total number particles in mask
    N_ref = x_ref.shape[0]

    # particle density rho, unit: number per nano meter square
    mask_area_nm2 = mask.area * (nm_per_pxl**2)
    rho_ref_per_nm2 = N_ref / mask_area_nm2

    # setup bins and ring areas
    bin_starts = np.arange(0, r_max_nm - ringwidth_nm, dr_slidingrings_nm)
    bin_ends = bin_starts + ringwidth_nm
    ring_areas_nm2 = np.pi * (
        bin_ends**2 - bin_starts**2
    )  # area of rings, unit nm square
    ring_areas_pxl2 = ring_areas_nm2 / (nm_per_pxl**2)

    # Calculate corrected histogram of distances
    lst_hist_per_point_auto_ref = []
    for i in range(len(x_ref)):
        # Calculate edge correction factor
        rings = [
            Point(x_ref[i], y_ref[i])
            .buffer(end)
            .difference(Point(x_ref[i], y_ref[i]).buffer(start))
            for start, end in zip(bin_starts / nm_per_pxl, bin_ends / nm_per_pxl)
        ]
        intersect_areas = np.array(
            [mask.intersection(Polygon(ring), grid_size=0.1).area for ring in rings]
        )
        edge_correction_factors = 1 / (intersect_areas / ring_areas_pxl2)

        # auto correlation - ref
        lst_hist = []
        for j in range(len(x_ref)):
            distance = (
                np.sqrt((x_ref[i] - x_ref[j]) ** 2 + (y_ref[i] - y_ref[j]) ** 2)
                * nm_per_pxl
            )
            lst_hist.append(((bin_starts <= distance) & (bin_ends >= distance)) * 1)
        hist_per_point_corrected = np.sum(lst_hist, axis=0) * edge_correction_factors
        lst_hist_per_point_auto_ref.append(hist_per_point_corrected)

    # calculate normalization factor that counts for density and ring area
    norm_factors_auto_ref = N_ref * ring_areas_nm2 * rho_ref_per_nm2

    PairCorr_auto_ref = (
        np.sum(lst_hist_per_point_auto_ref, axis=0) / norm_factors_auto_ref
    )

    return PairCorr_auto_ref


# Function to process a single file
def process_file(
    i,
    lst_fname_csv,
    lst_fname_PAINT,
    nm_per_pxl,
    r_max_nm,
    ringwidth_nm,
    dr_slidingrings_nm,
):
    df_import = pd.read_csv(lst_fname_csv[i])
    img_PAINT = imread(lst_fname_PAINT[i])

    img_denoise = gaussian_filter(img_PAINT, sigma=1)
    edges = img_denoise >= 3
    contours, _ = cv2.findContours(edges * 1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    mask = cnt2mask(edges.shape, contours)
    contours_final, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    # get only the major condensate
    cnt_condensate = contours_final[0]
    for cnt in contours_final:
        if cv2.contourArea(cnt) > cv2.contourArea(cnt_condensate):
            cnt_condensate = cnt

    mask = Polygon(np.squeeze(cnt_condensate))
    mask = erode_polygon(mask)

    df = filter_perTrack(df_import)

    auto_PCF = PairCorr_with_edge_correction(
        df,
        mask,
        nm_per_pxl,
        r_max_nm,
        ringwidth_nm,
        dr_slidingrings_nm,
    )

    return (
        auto_PCF,
        df.shape[0],
    )


def main():
    print(
        "Choose all tif and csv files to be batch proccessed. The prefix should be the same for (1) .csv (2) -PAINT.tif"
    )
    lst_files = list(fd.askopenfilenames())
    folder_data = dirname(lst_files[0])
    os.chdir(folder_data)
    fname_save = "PairCorr-erode5-DataDict-pooled-perTrack.p"

    # Parameters
    nm_per_pxl = 117  # ONI scale
    r_max_nm = 1120
    ringwidth_nm = 100
    dr_slidingrings_nm = 20  # stepsize between adjascent overlaping rings, nm
    bins = np.arange(
        0, r_max_nm - ringwidth_nm, dr_slidingrings_nm
    )  # overlaping bins (sliding window)

    all_files = os.listdir(".")
    lst_fname_csv = [f for f in all_files if f.endswith(".csv")]
    lst_fname_PAINT = [f.split(".csv")[0] + "-PAINT.tif" for f in lst_fname_csv]

    # The tqdm library provides an easy way to visualize the progress of loops.
    pbar = tqdm(total=len(lst_fname_csv))

    # Update function for the progress bar
    def update(*a):
        pbar.update()

    # Create a process pool and map the function to the files
    with Pool(cpu_count()) as p:
        results = []
        for i in range(len(lst_fname_csv)):
            result = p.apply_async(
                process_file,
                args=(
                    i,
                    lst_fname_csv,
                    lst_fname_PAINT,
                    nm_per_pxl,
                    r_max_nm,
                    ringwidth_nm,
                    dr_slidingrings_nm,
                ),
                callback=update,
            )
            results.append(result)

        # Wait for all processes to finish and gather results
        results = [r.get() for r in results]
    pbar.close()

    # Unpack results into separate lists
    (
        lst_auto_PCF,
        lst_size,
    ) = map(list, zip(*results))

    dict_to_save = {
        "filenames": lst_fname_csv,
        "lst_N_locations": lst_size,
        "lst_auto_PCF": lst_auto_PCF,
        "nm_per_pxl": nm_per_pxl,
        "r_max_nm": r_max_nm,
        "ringwidth_nm": ringwidth_nm,
        "dr_slidingrings_nm": dr_slidingrings_nm,
        "bins": bins,
    }
    pickle.dump(
        dict_to_save,
        open(join(folder_data, fname_save), "wb"),
    )
    print("Saved successfully at the following path:")
    print(join(folder_data, fname_save))


if __name__ == "__main__":
    main()
