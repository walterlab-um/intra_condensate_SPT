import os
import scipy.stats as stats
from shapely.geometry import Point, Polygon
import numpy as np
import pickle
import pandas as pd
from tkinter import filedialog as fd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'


###############################################
# Setting parameters
# path
print("Choose the RNA track csv files for processing:")
lst_files = list(fd.askopenfilenames())
# scalling factors for physical units
um_per_pixel = 0.117


###############################################
# Functions
def distance_to_edge(df_in, um_per_pixel, cnt_coordinates):
    """
    Pool step sizes from all >=7 steps RNA tracks within condensates, also calculate their distance to edge.
    """
    # get all track id
    trackids = df_in.trackID.unique()

    lst_midx = []  # pixel
    lst_midy = []  # pixel
    lst_stepsize = []  # nm
    lst_conR = []  # condensate size, radius, nm
    lst_step2edge = []  # minimal distance to boundary, nm

    # iteration through all tracks
    id = trackids[0]
    for id in trackids:
        df_track = df_in[df_in.trackID == id]

        # check track length
        if df_track.t.max() - df_track.t.min() < 6:  # skip short tracks
            continue

        # check if RNA is within any contours, and save the contour
        within_condensate = False
        for cnt in cnt_coordinates:
            cnt_polygon = Polygon(cnt)
            point_RNA = Point(df_track.x.mean(), df_track.y.mean())
            if point_RNA.within(cnt_polygon):
                within_condensate = True
                cnt_hit = cnt_polygon
        if within_condensate is False:
            continue

        # condensate size
        condensate_R = np.sqrt(4 * cnt_hit.area / np.pi) / 2 * um_per_pixel * 1000
        # step sizes
        singleRNA_stepsize = list(
            np.sqrt(
                (df_track.x[1:].to_numpy() - df_track.x[:-1].to_numpy()) ** 2
                + (df_track.y[1:].to_numpy() - df_track.y[:-1].to_numpy()) ** 2
            )
            * um_per_pixel
            * 1000
        )
        # middle point of each step
        singleRNA_mid_x = (df_track.x[:-1].to_numpy() + df_track.x[1:].to_numpy()) / 2
        singleRNA_mid_y = (df_track.y[:-1].to_numpy() + df_track.y[1:].to_numpy()) / 2
        # step distance to edge
        singleRNA_step2edge = []
        for midx, midy in zip(singleRNA_mid_x, singleRNA_mid_y):
            singleRNA_step2edge.append(
                cnt_hit.exterior.distance(Point(midx, midy)) * um_per_pixel * 1000
            )
        # assign values
        lst_stepsize.extend(singleRNA_stepsize)
        lst_step2edge.extend(singleRNA_step2edge)
        lst_conR.extend(list(np.repeat(condensate_R, len(singleRNA_stepsize))))

    df_out = pd.DataFrame(
        {
            "stepsize": lst_stepsize,
            "step-edge distance (nm)": lst_step2edge,
            "condensate R (nm)": lst_conR,
        }
    )

    return df_out


###############################################
# Main body
# f = "/Volumes/AnalysisGG/PROCESSED_DATA/2022July-RNAinFUS-preliminary/20220712_FLmRNA_10FUS_1Mg_10Dex_noTotR_24C/low_freq_05Hz_FOV-RNA.csv"
for f in track(lst_files):
    df_in = pd.read_csv(f, dtype=float)
    df_in = df_in.astype({"t": int})
    fname_pkl = f.strip("-RNA.csv") + "-condensate_ilastik_contours.pkl"

    contours, _, _ = pickle.load(open(fname_pkl, "rb"))
    cnt_coordinates = []
    for cnt in contours:
        cnt_coordinates.append([tuple(cnt[i, 0]) for i in range(cnt.shape[0])])

    df_out = distance_to_edge(df_in, um_per_pixel, cnt_coordinates)
    fname_save = f.strip(".csv") + "_steps.csv"
    df_out.to_csv(fname_save)
