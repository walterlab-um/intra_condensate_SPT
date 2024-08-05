from tifffile import imread, imwrite
from scipy.ndimage import gaussian_filter
from tkinter import filedialog as fd
import numpy as np
from rich.progress import track

print("Choose all tif files to be batch proccessed:")
lst_files = list(fd.askopenfilenames())
sigma = 1


###############################
for f in track(lst_files):
    video = imread(f)
    video_out = []
    for img in video:
        img_filtered = gaussian_filter(img, sigma=1)
        video_out.append(img_filtered)
    video_out = np.stack(video_out)
    fsave = f[:-4] + "-smoothed.tif"
    if len(video.shape) < 3:  # image:
        imwrite(fsave, video_out)
    elif len(video.shape) > 2:  # video:
        imwrite(fsave, video_out, metadata={"axes": "TYX"})
