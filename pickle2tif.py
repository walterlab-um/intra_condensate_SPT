from tifffile import imwrite
from skimage.util import img_as_uint
from tkinter import filedialog as fd
import os
from os.path import dirname, basename
import numpy as np
import pandas as pd
from rich.progress import track
import pickle

pd.options.mode.chained_assignment = None  # default='warn'

print("Choose all '_phys_unit.p' files for processing:")
lst_path = list(fd.askopenfilenames())

folder_save = dirname(lst_path[0])
os.chdir(folder_save)
lst_files = [basename(f) for f in lst_path]

for fpath in lst_files:
    fname = basename(fpath)
    img = pickle.load(open(fname, "rb"))
    imwrite(
        fname[:-2] + ".tif",
        img,
        # imagej=True,
        metadata={"um_per_pixel_PAINT": 0.117},
    )
