import os
from os.path import dirname, basename
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely import distance
from rich.progress import track
from tkinter import filedialog as fd

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format
# This script gives the collocalization information for every RNA position over time.
# This version is for in vitro RNA SPT in reconstituted condensates
# Update: enable user to choose list of SPT and condensate AIO files. The filename format should be the following. The middle part is used for matching the two.
# condensates_AIO-20221031-FL_noTR_noDex_20ms_0hr_Replicate1_FOV-1-condensates_AveProj_Simple Segmentation.csv
# SPT_results_AIO-20221031-FL_noTR_noDex_20ms_0hr_Replicate1_FOV-1-RNAs.csv
# Results will be saved in the RNA's folder


# scalling factors for physical units
nm_per_pixel = 117
print("Scaling factors: nm_per_pixel = " + str(nm_per_pixel))

print("Choose all SPT_results_AIO-xxxxx-RNAs.csv files:")
all_fpath_RNA_AIO = list(fd.askopenfilenames())
print(
    "Choose all condensates_AIO-xxxxx-condensates_AveProj_Simple Segmentation.csv files:"
)
all_fpath_condensate_AIO = list(fd.askopenfilenames())
os.chdir(dirname(all_fpath_RNA_AIO[0]))

# Output file columns
columns = [
    "fname_RNA",
    "fname_condensate",
    "RNA_trackID",
    "t",
    "x",
    "y",
    "InCondensate",
    "condensateID",
    "R_nm",
    "distance_to_center_nm",
    "distance_to_edge_nm",
]


def list_like_string_to_polygon(list_like_string):
    # example list_like_string structure of polygon coordinates: '[[196, 672], [196, 673], [197, 673], [198, 673], [199, 673], [199, 672], [198, 672], [197, 672]]'
    list_of_xy_string = list_like_string[2:-2].split("], [")

    coords_roi = []
    for xy_string in list_of_xy_string:
        x, y = xy_string.split(", ")
        coords_roi.append((int(x), int(y)))

    polygon_output = Polygon(coords_roi)

    return polygon_output


def list_like_string_to_xyt(list_like_string):
    # example list_like_string structure of xyt: '[0, 1, 2, 3]'
    list_of_xyt_string = list_like_string[1:-1].split(", ")
    lst_xyt = []
    for xyt_string in list_of_xyt_string:
        lst_xyt.append(float(xyt_string))

    return lst_xyt


## loop through each FOV
for fpath_RNA_AIO in track(all_fpath_RNA_AIO):
    # find the matching condensate_AIO file path
    fname_RNA_AIO = basename(fpath_RNA_AIO)
    experiment_descriptions = fname_RNA_AIO[16:-9]
    fpath_condensate_AIO = None
    for path in all_fpath_condensate_AIO:
        if experiment_descriptions + "-condensates" in path:
            fpath_condensate_AIO = path
            fname_condensate_AIO = basename(path)
    if fpath_condensate_AIO is None:
        print("No match for:", fname_RNA_AIO)
        continue

    # load the pair of AIO files for the current FOV
    df_RNA = pd.read_csv(fpath_RNA_AIO)
    df_condensate = pd.read_csv(fpath_condensate_AIO)

    ## process RNA tracks one by one
    lst_rows_of_df = []
    # load condensates near the RNA as dictionary of polygons
    for trackID in df_RNA["trackID"].unique():
        current_track = df_RNA[df_RNA["trackID"] == trackID]
        lst_x = list_like_string_to_xyt(current_track["list_of_x"].squeeze())
        lst_y = list_like_string_to_xyt(current_track["list_of_y"].squeeze())
        lst_t = list_like_string_to_xyt(current_track["list_of_t"].squeeze())
        mean_RNA_x = np.mean(lst_x)
        mean_RNA_y = np.mean(lst_y)

        # to save computation time, only search for condensates near the RNA
        all_condensateID_nearby = []
        for _, row in df_condensate.iterrows():
            center_x_pxl = row["center_x_pxl"]
            center_y_pxl = row["center_y_pxl"]
            if (center_x_pxl - mean_RNA_x) ** 2 + (
                center_y_pxl - mean_RNA_y
            ) ** 2 > 50**2:
                continue
            else:
                all_condensateID_nearby.append(row["condensateID"])

        dict_condensate_polygons_nearby = dict()
        for condensateID_nearby in all_condensateID_nearby:
            str_condensate_coords = df_condensate[
                df_condensate["condensateID"] == condensateID_nearby
            ]["contour_coord"].squeeze()

            lst_tup_condensate_coords = []
            for str_condensate_xy in str_condensate_coords[2:-2].split("], ["):
                condensate_x, condensate_y = str_condensate_xy.split(", ")
                lst_tup_condensate_coords.append(
                    tuple([int(condensate_x), int(condensate_y)])
                )

            dict_condensate_polygons_nearby[condensateID_nearby] = Polygon(
                lst_tup_condensate_coords
            )

        # process each position in track one by one
        for i in range(len(lst_t)):
            t = lst_t[i]
            x = lst_x[i]
            y = lst_y[i]

            point_RNA = Point(x, y)

            ## Perform colocalization
            # search for which condensate it's in
            InCondensate = False
            for key, polygon in dict_condensate_polygons_nearby.items():
                # if point_RNA.within(polygon):
                if (
                    distance(point_RNA, polygon) < 2
                ):  # redundancy: to include RNAs near boundary
                    InCondensate = True
                    condensateID = key
                    R_nm = np.sqrt(polygon.area * nm_per_pixel**2 / np.pi)
                    # p1, p2 = nearest_points(polygon, point_RNA)
                    distance_to_edge_nm = (
                        polygon.exterior.distance(point_RNA) * nm_per_pixel
                    )
                    distance_to_center_nm = (
                        distance(polygon.centroid, point_RNA) * nm_per_pixel
                    )
                    break
            if not InCondensate:
                condensateID = np.nan
                R_nm = np.nan
                distance_to_center_nm = np.nan
                distance_to_edge_nm = np.nan

            # Save
            new_row = [
                fname_RNA_AIO,
                fname_condensate_AIO,
                trackID,
                t,
                x,
                y,
                InCondensate,
                condensateID,
                R_nm,
                distance_to_center_nm,
                distance_to_edge_nm,
            ]
            lst_rows_of_df.append(new_row)

    df_save = pd.DataFrame.from_records(
        lst_rows_of_df,
        columns=columns,
    )
    fname_save = "colocalization_AIO-" + experiment_descriptions + ".csv"

    df_save.to_csv(fname_save, index=False)
