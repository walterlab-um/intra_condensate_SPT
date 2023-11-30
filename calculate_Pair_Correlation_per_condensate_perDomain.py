from shapely.geometry import Point, Polygon
from skimage.filters import threshold_otsu
from skimage.filters import laplace
from skimage.feature import blob_log
from scipy.ndimage import gaussian_filter
from tifffile import imread
import cv2
import math
import os
from os.path import join, dirname
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
import matplotlib.colors as clr

pd.options.mode.chained_assignment = None  # default='warn'


"""
The program calculates per-domain (FUS) and per-location (RNA) Pair Correlation Function (PCF; aka. Radial Distribution Function, RDF) of two channel single partical tracking (SPT) data.

Per-domain means a Laplacian pyramid method is first used to find FUS nano domains of various sizes (thus various scales in the pyramid)

Per-location means every single location within any trajectory in the SPT dataset is treated as a single point. Such point set is then subjected to PCF calculation. To avoid long trajectories contributing to too many points and thus dominates the point set. Only 10 locations of any tracks longer than 10 steps are taken (via random sampling).

It saves four PCF with corresponding informatioon in a single pickle file:
1. auto PCF of channel 1 (FUS)
2. auto PCF of channel 2 (RNA)
3. cross PCF using channel 1 (FUS) spots as reference
4. cross PCF using channel 2 (RNA) spots as reference, which is theoretically equivalent to 3 and is provided as a fact check.
"""

print(
    "Choose all tif and csv files to be batch proccessed. The prefix should be the same for (1) left.csv (2) right.csv (3) left-PAINT.tif (4) right-PAINT.tif"
)
lst_files = list(fd.askopenfilenames())
folder_data = dirname(lst_files[0])
os.chdir(folder_data)
folder_save = "../"
fname_save = "PairCorr-DataDict-pooled-perDomain.p"

# Parameters
nm_per_pxl = 117  # ONI scale
r_max_nm = 1120
ringwidth_nm = 100
dr_slidingrings_nm = 20  # stepsize between adjascent overlaping rings, nm
bins = np.arange(
    0, r_max_nm - ringwidth_nm, dr_slidingrings_nm
)  # overlaping bins (sliding window)
# round UP for boundary correction, see function PairCorrelation_perCell()
r_max_pxl = math.ceil(r_max_nm / 117)
cmap_dark2cyan = clr.LinearSegmentedColormap.from_list(
    "dark2cyan",
    ["black", "#B9DBF4"],
    N=200,
)

all_files = os.listdir(".")
lst_fname_left_csv = [f for f in all_files if f.endswith("left.csv")]
lst_fname_left_PAINT = [
    f.split("left.csv")[0] + "left-PAINT.tif" for f in lst_fname_left_csv
]
lst_fname_right_csv = [f.split("left.csv")[0] + "right.csv" for f in lst_fname_left_csv]
lst_fname_right_PAINT = [
    f.split("left.csv")[0] + "right-PAINT.tif" for f in lst_fname_left_csv
]


def cnt2mask(imgshape, contours):
    # create empty image
    mask = np.zeros(imgshape, dtype=np.uint8)
    # draw contour
    for cnt in contours:
        cv2.fillPoly(mask, [cnt], (255))
    return mask


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


def detect_domains(img_PAINT_left, center_xy, mask_r):
    log_image = laplace(gaussian_filter(img_PAINT_left, 1))
    threshold = math.ceil(threshold_otsu(np.abs(log_image)))
    domains = blob_log(
        img_PAINT_left,
        min_sigma=1,
        max_sigma=3,
        threshold=threshold,
    )

    # each row in `blobs` is (y,x,sigma) where (y,x) are coordinates of the blob and sigma is the standard deviation of the Gaussian kernel which detected the blob.
    domains = np.stack(domains)
    domains_x = domains[:, 1]
    domains_y = domains[:, 0]

    # Finally, converting sigma values to approximate radius values, since log scale space was used, the radius of blobs would be sqrt(2)*sigma
    domains_r = domains[:, 2] * np.sqrt(2)

    d2center = np.sqrt(
        (domains_x - center_xy[0]) ** 2 + (domains_y - center_xy[1]) ** 2
    )
    d2center_norm = d2center / mask_r

    df = pd.DataFrame(
        {
            "x": domains_x,
            "y": domains_y,
            "r": domains_r,
            "d2center": d2center,
            "condensate_r": np.repeat(mask_r, len(domains_x)),
            "d2center_norm": d2center_norm,
        },
        dtype=float,
    )

    return df


def PairCorr_with_edge_correction(df_ref, df_interest, mask):
    # only count particles within mask
    x_ref, y_ref = corr_within_mask(df_ref, mask)
    x_interest, y_interest = corr_within_mask(df_interest, mask)

    # Total number particles in mask
    N_ref = x_ref.shape[0]
    N_interest = x_interest.shape[0]

    # particle density rho, unit: number per nano meter square
    mask_area_nm2 = mask.area * (nm_per_pxl**2)
    rho_ref_per_nm2 = N_ref / mask_area_nm2
    rho_interest_per_nm2 = N_interest / mask_area_nm2

    # setup bins and ring areas
    bin_starts = np.arange(0, r_max_nm - ringwidth_nm, dr_slidingrings_nm)
    bin_ends = bin_starts + ringwidth_nm
    ring_areas_nm2 = np.pi * (
        bin_ends**2 - bin_starts**2
    )  # area of rings, unit nm square
    ring_areas_pxl2 = ring_areas_nm2 / (nm_per_pxl**2)

    # Calculate corrected histogram of distances
    lst_hist_per_point_cross = []
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

        # cross correlation
        lst_hist = []
        for j in range(len(x_interest)):
            distance = (
                np.sqrt(
                    (x_ref[i] - x_interest[j]) ** 2 + (y_ref[i] - y_interest[j]) ** 2
                )
                * nm_per_pxl
            )
            lst_hist.append(((bin_starts <= distance) & (bin_ends >= distance)) * 1)
        hist_per_point_corrected = np.sum(lst_hist, axis=0) * edge_correction_factors
        lst_hist_per_point_cross.append(hist_per_point_corrected)

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
    norm_factors_cross = N_ref * ring_areas_nm2 * rho_interest_per_nm2
    norm_factors_auto_ref = N_ref * ring_areas_nm2 * rho_ref_per_nm2

    PairCorr_cross = np.sum(lst_hist_per_point_cross, axis=0) / norm_factors_cross
    PairCorr_auto_ref = (
        np.sum(lst_hist_per_point_auto_ref, axis=0) / norm_factors_auto_ref
    )

    return PairCorr_cross, PairCorr_auto_ref


# Function to process a single file
def process_file(i):
    # df_left = pd.read_csv(lst_fname_left_csv[i])
    df_right = pd.read_csv(lst_fname_right_csv[i])
    img_PAINT_left = imread(lst_fname_left_PAINT[i])
    img_PAINT_right = imread(lst_fname_right_PAINT[i])

    img_denoise = gaussian_filter(img_PAINT_left + img_PAINT_right, sigma=1)
    edges = img_denoise >= 10
    # edges = (img_PAINT_left > 5) & (img_PAINT_right > 5)
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

    df_left = detect_domains(img_PAINT_left, center_xy, mask_r)
    df_right = filter_perLoc(df_right)

    cross_FUSref, auto_FUS = PairCorr_with_edge_correction(df_left, df_right, mask)
    # cross_RNAref, auto_RNA = PairCorr_with_edge_correction(df_right, df_left, mask)

    # Plot the detected domains
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(img_PAINT_left, cmap=cmap_dark2cyan)
    plt.plot(
        [2, 2 + 1000 / nm_per_pxl],
        [2, 2],
        "-",
        color="w",
        lw=5,
    )
    ax.invert_yaxis()
    plt.axis("off")
    plt.axis("equal")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("#PAINT location", size=17)
    cbar.ax.yaxis.set_tick_params(labelsize=17)
    cbar.ax.yaxis.set_label_position("left")
    for x, y, r in zip(df_left.x, df_left.y, df_left.r):
        c = plt.Circle((x, y), r, color="w", fill=False)
        ax.add_patch(c)
    plt.savefig(
        lst_fname_left_csv[i][:-9] + "-domains.png",
        format="png",
        dpi=300,
        bbox_inches="tight",
    )

    return (
        cross_FUSref,
        auto_FUS,
        # cross_RNAref,
        # auto_RNA,
        df_left.shape[0],
        df_right.shape[0],
        df_left,  # save all FUS domains (x, y, r, d2center, condensate_r, d2center_norm)
    )


def main():
    # The tqdm library provides an easy way to visualize the progress of loops.
    pbar = tqdm(total=len(lst_fname_left_csv))

    # Update function for the progress bar
    def update(*a):
        pbar.update()

    # Create a process pool and map the function to the files
    with Pool(cpu_count()) as p:
        results = []
        for i in range(len(lst_fname_left_csv)):
            result = p.apply_async(process_file, args=(i,), callback=update)
            results.append(result)

        # Wait for all processes to finish and gather results
        results = [r.get() for r in results]
    pbar.close()

    # Unpack results into separate lists
    (
        lst_cross_FUSref,
        lst_autoFUS,
        # lst_cross_RNAref,
        # lst_autoRNA,
        lst_size_FUS,
        lst_size_RNA,
        lst_df_FUSdomains,
    ) = map(list, zip(*results))

    dict_to_save = {
        "filenames_FUS": lst_fname_left_csv,
        "filenames_RNA": lst_fname_right_csv,
        "lst_N_locations_FUS": lst_size_FUS,
        "lst_N_locations_RNA": lst_size_RNA,
        "lst_cross_PCF_FUSref": lst_cross_FUSref,
        # "lst_cross_PCF_RNAref": lst_cross_RNAref,
        "lst_auto_PCF_FUS": lst_autoFUS,
        # "lst_auto_PCF_RNA": lst_autoRNA,
        "lst_df_FUSdomains": lst_df_FUSdomains,  # save all FUS domains (x, y, r, d2center, condensate_r, d2center_norm)
        "nm_per_pxl": nm_per_pxl,
        "r_max_nm": r_max_nm,
        "ringwidth_nm": ringwidth_nm,
        "dr_slidingrings_nm": dr_slidingrings_nm,
        "bins": bins,
    }
    pickle.dump(
        dict_to_save,
        open(join(folder_save, fname_save), "wb"),
    )
    print("Saved successfully at the following path:")
    print(join(folder_save, fname_save))


if __name__ == "__main__":
    main()
