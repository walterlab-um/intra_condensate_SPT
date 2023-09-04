import cv2
from tifffile import imread, imwrite
import pickle
import numpy as np
from tkinter import filedialog as fd
from rich.progress import track


#########################################
# Load and organize files

print("Type 1 to keep left, 2 to keep right channel")
selector = input()

if (selector != "1") & (selector != "2"):
    print("Please type only 1 or 2")
    exit()

print("Choose the tif files for crop")
lst_files = list(fd.askopenfilenames())


#########################################
# Apply registration and Crop
for fpath in track(lst_files):
    # load the tiff file
    video = imread(fpath)
    halfwidth = int(video.shape[2] / 2)
    frames = int(video.shape[0])

    # split left and right
    video_left = video[:, :, 0:halfwidth]
    video_right = video[:, :, halfwidth:]

    if selector == "1":
        video_out = video_left
    elif selector == "2":
        video_out = video_right

    fsave_right = fpath[-4] + "-cropped.tif"
    imwrite(
        fsave_right,
        video_out,
        imagej=True,
        metadata={"axes": "TYX"},
    )
