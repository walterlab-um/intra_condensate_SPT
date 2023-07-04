import os
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from shapely import distance
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

# AIO: All in one format
# This script gives the collocalization information for every RNA position over time.
# This version is for in vitro RNA SPT in reconstituted condensates


# scalling factors for physical units
nm_per_pixel = 117
print("Scaling factors: nm_per_pixel = " + str(nm_per_pixel))

# Change the below directory to a mother folder containing subfolders for each condition. Within each subfolder, a "condensate" folder must be found containing all condensate videos in tif format.
dir_from = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA-diffusion-in-FUS/RNA_SPT_in_FUS-May2023_wrapup"


# Output file columns
columns = [
    "fname_FOV",
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


os.chdir(dir_from)
all_fname_RNA_AIO = [f for f in os.listdir(".") if f.startswith("SPT_results_AIO")]
all_fname_condensate_AIO = [
    f for f in os.listdir(".") if f.startswith("condensates_AIO")
]


## loop through each condition
for fname_RNA_AIO in all_fname_RNA_AIO:
    df_RNA = pd.read_csv(fname_RNA_AIO)
    df_condensate = pd.read_csv(
        "condensates_AIO" + fname_RNA_AIO.split("SPT_results_AIO")[-1]
    )

    ## process RNA tracks one by one
    lst_rows_of_df = []
    for fname_FOV in track(df_RNA["filename"].unique(), description=fname_RNA_AIO):
        df_RNA_current_FOV = df_RNA[df_RNA["filename"] == fname_FOV]
        # load condensates near the RNA as dictionary of polygons
        df_condensate_current_FOV = df_condensate[
            df_condensate["filename"].str.startswith(fname_FOV[:-27] + "condensates")
        ]
        for trackID in df_RNA_current_FOV["trackID"].unique():
            current_track = df_RNA_current_FOV[df_RNA_current_FOV["trackID"] == trackID]
            lst_x = list_like_string_to_xyt(current_track["list_of_x"].squeeze())
            lst_y = list_like_string_to_xyt(current_track["list_of_y"].squeeze())
            lst_t = list_like_string_to_xyt(current_track["list_of_t"].squeeze())
            mean_RNA_x = np.mean(lst_x)
            mean_RNA_y = np.mean(lst_y)

            # to save computation time, only search for condensates near the RNA
            all_condensateID_nearby = []
            for _, row in df_condensate_current_FOV.iterrows():
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
                str_condensate_coords = df_condensate_current_FOV[
                    df_condensate_current_FOV["condensateID"] == condensateID_nearby
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
                    if point_RNA.within(polygon):
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
                    fname_FOV,
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
    fname_save = "colocalization_AIO-" + fname_RNA_AIO.split("AIO-")[-1]

    df_save.to_csv(fname_save, index=False)
