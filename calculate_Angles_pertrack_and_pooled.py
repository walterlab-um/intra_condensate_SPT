import os
import pandas as pd
import numpy as np
from copy import deepcopy
from rich.progress import track


# Output file each row has elements: trackID, N_steps (could be used to filter out short tracks, whcih would bias), total displacement (um), the list of angles, fraction of different bins.
# fractions are summed to 1; fraction = density * bin width
# The bins can be altered below
bins = np.linspace(0, 180, 10).astype(int)  # #boundaries = #bins + 1
lst_fraction_titles = [
    "(" + str(bins[i]) + "," + str(bins[i + 1]) + "]" for i in range(len(bins) - 1)
]
columns = [
    "trackID",
    "N_steps",
    "total displacement (um)",
    "list of angles",
] + lst_fraction_titles

s_per_frame = 0.02
um_per_pixel = 0.117
folder_path = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup/Verify_alpha-angledist_on_mobile_vs_constrained"
os.chdir(folder_path)

lst_fname = [f for f in os.listdir(folder_path) if f.startswith("Constrained_")]


def install_new_trackID(df_in):
    current_t = 0
    trackID = 0
    lst_newID = []
    for t_in_track in df_in["t"]:
        if t_in_track >= current_t:
            lst_newID.append(trackID)
            current_t += 1
        elif t_in_track < current_t:
            trackID += 1
            lst_newID.append(trackID)
            current_t = 1
    df_out = deepcopy(df_in)
    df_out["trackID"] = lst_newID
    return df_out


def calc_angle(x, y):
    # x and y at time 0 and time 1
    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]
    # unit vectors of all steps, and step 0 and step 1
    vector = np.array([x1 - x0, y1 - y0])
    # convert to complex number to use np.angle
    vector_complex = 1j * vector[1, :]
    vector_complex += vector[0, :]
    angles_eachstep = np.angle(vector_complex, deg=True)
    angles = np.ediff1d(angles_eachstep)  # between adjacent steps
    # convert all angles to within range (0,+-180) for output
    angles[angles < -180] = angles[angles < -180] + 360
    angles[angles > 180] = angles[angles > 180] - 360

    return angles


for fname in lst_fname:
    df_tracks = pd.read_csv(fname, dtype=float)
    df_tracks = install_new_trackID(df_tracks)
    lst_rows_of_df = []

    for trackID in track(df_tracks["trackID"].unique(), description=fname):
        df_current_track = df_tracks[df_tracks["trackID"] == trackID]
        x = df_current_track["x"].to_numpy(dtype=float)
        y = df_current_track["y"].to_numpy(dtype=float)

        N_steps = x.shape[0]
        total_disp = (
            np.sum(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)) * um_per_pixel
        )

        angles = calc_angle(x, y)
        densities, _ = np.histogram(np.absolute(angles), bins, density=True)
        # fractions are summed to 1; fraction = density * bin width
        fractions = densities * (bins[1] - bins[0])

        # each row has elements: trackID, N_steps, total displacement (um), the list of angles, fractions.
        new_row = [trackID, N_steps, total_disp, angles.tolist()]
        new_row.extend(fractions.tolist())
        lst_rows_of_df.append(new_row)

    df_save = pd.DataFrame.from_records(
        lst_rows_of_df,
        columns=columns,
    )
    fname_save = "Angles_" + fname
    df_save.to_csv(fname_save, index=False)
