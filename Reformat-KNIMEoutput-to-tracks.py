import pandas as pd
import numpy as np
from os.path import dirname, join
from tkinter import filedialog as fd
from rich.progress import track

print("Please choose the single csv file from KNIME:")
path = fd.askopenfilename()
print("Now processing:", path)
df_in = pd.read_csv(path)


def filter_length_disp(df, min_length=5):
    df = df.sort_values(by=["trackID", "t"])

    lst_trackID = []
    lst_t = []
    lst_x = []
    lst_y = []
    lst_meanInt = []
    lst_estD = []
    lst_contrast = []

    # trackID = df.trackID.unique().astype("int64")[0]
    for trackID in df.trackID.unique().astype("int64"):
        df_track = df[df.trackID == trackID]

        # filter out short tracks
        # the given the gaps, the real track length should be max t - min t
        tracklength = df_track["t"].max() - df_track["t"].min()
        if tracklength >= min_length:
            lst_trackID.extend(list(np.repeat(trackID, df_track.shape[0])))
            lst_t.extend(list(df_track["t"]))
            lst_x.extend(list(df_track["x"]))
            lst_y.extend(list(df_track["y"]))
            lst_meanInt.extend(list(df_track["meanIntensity"]))
            lst_estD.extend(list(df_track["estDiameter"]))
            lst_contrast.extend(list(df_track["contrast"]))

    df_out = pd.DataFrame(
        {
            "trackID": lst_trackID,
            "t": lst_t,
            "x": lst_x,
            "y": lst_y,
            "meanIntensity": lst_meanInt,
            "estDiameter": lst_estD,
            "contrast": lst_contrast,
        },
        dtype=float,
    )

    return df_out


# each row is one cell. iterate and format each row into individual files
for idx in track(range(df_in.shape[0])):
    row = df_in.iloc[idx]
    fname_save = row["row ID"].split(".")[0] + "_reformatted.csv"
    path_save = join(dirname(path), fname_save)
    df_formatted = pd.DataFrame(
        {
            "spotID": row["spotID"][1:-1].split(","),
            "trackID": row["trackID"][1:-1].split(","),
            "t": row["frame"][1:-1].split(","),
            "x": row["x"][1:-1].split(","),
            "y": row["y"][1:-1].split(","),
            "meanIntensity": row["meanIntensity"][1:-1].split(","),
            "estDiameter": row["estDiameter"][1:-1].split(","),
            "contrast": row["contrast"][1:-1].split(","),
        },
        dtype=float,
    )
    df_out = filter_length_disp(df_formatted)
    df_out.to_csv(path_save, index=False)
