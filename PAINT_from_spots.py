from tifffile import imwrite
from skimage.util import img_as_uint
import os
from os.path import dirname, basename
from tkinter import filedialog as fd
import numpy as np
import pandas as pd
from rich.progress import track

pd.options.mode.chained_assignment = None  # default='warn'

print("Choose all spots-reformatted.csv files to be batch proccessed into SMLM tiff:")
lst_files = list(fd.askopenfilenames())

um_per_pixel = 0.117
scaling_factor = 1
um_per_pixel_PAINT = um_per_pixel / scaling_factor
xpixels_ONI = 418
ypixels_ONI = 674
xedges = np.arange((xpixels_ONI + 1) * scaling_factor)
yedges = np.arange((ypixels_ONI + 1) * scaling_factor)

folder_save = dirname(lst_files[0])
os.chdir(folder_save)
print("Now in folder:", folder_save)

for fpath in track(lst_files):
    fname = basename(fpath)
    df = pd.read_csv(fname)

    img_PAINT, _, _ = np.histogram2d(
        x=df["x"] * scaling_factor,
        y=df["y"] * scaling_factor,
        bins=(xedges, yedges),
    )

    imwrite(
        fname.split("-spot")[0] + "-SMLM.tif",
        img_PAINT.astype("uint8"),
        imagej=True,
        metadata={"um_per_pixel_PAINT": um_per_pixel_PAINT},
    )
