import cv2
from tifffile import imread, imwrite
import pickle
import numpy as np
from tkinter import filedialog as fd
from rich.progress import track

# In vitro condensate-RNA tracking: image RNA first for the majority of the video (i.e. 200 frames) then image condensates for the last 10 frames. The code extracts the maximum projection of the last ten frames of condensates and duplicate it to a full video of 200 frames so they can overlay by imageJ.

#########################################
# Load and organize files
# ONI
print("Pick the channel to perform transformation, 1 for left, 2 for right:")
selector = input()
if selector == "1":
    print("Choose the matrix for left")
elif selector == "2":
    print("Choose the matrix for left")
else:
    print("Please enter either 1 or 2")
    exit()
path_matrix = fd.askopenfilename()
warp_matrix = pickle.load(open(path_matrix, "rb"))

print("Choose the tif files for channel alignment:")
lst_files = list(fd.askopenfilenames())


def crop_imgstack(imgstack):
    # dimension of the imgstack should be (z,h,w)
    # this will crop a 5 pixel frame from the image
    z, h, w = imgstack.shape
    imgstack_out = imgstack[:, 10 : h - 10, 10 : w - 10]
    return imgstack_out


def transform(img2d, warp_matrix):
    sz = img2d.shape
    img_out = cv2.warpPerspective(
        img2d,
        warp_matrix,
        (sz[1], sz[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return img_out


#########################################
# Apply registration and Crop
for fpath in track(lst_files):
    # load the tiff file
    img = imread(fpath)
    halfwidth = int(img.shape[2] / 2)
    frames = int(img.shape[0])

    # split left and right
    img_left = img[:, :, 0:halfwidth]
    img_right = img[:, :, halfwidth:]

    if selector == "1":
        # Use warpPerspective for Homography transform ON EACH Z FRAME
        lst_img_left_aligned = [
            transform(img_left[z, :, :], warp_matrix) for z in range(img.shape[0])
        ]
        img_left_aligned = np.stack(lst_img_left_aligned, axis=0)
        img_right_aligned = img_right
    elif selector == "2":
        # Use warpPerspective for Homography transform ON EACH Z FRAME
        lst_img_right_aligned = [
            transform(img_right[z, :, :], warp_matrix) for z in range(img.shape[0])
        ]
        img_right_aligned = np.stack(lst_img_right_aligned, axis=0)
        img_left_aligned = img_left

    # again need odd for right, even for left
    # the left: crop, take last 10 frames, average projection, DO NOT duplicate
    img_left_cropped = crop_imgstack(img_left_aligned)
    img_left_aveproj = np.nanmean(img_left_cropped[-10:, :, :], axis=0)
    fsave_left = fpath.strip(".tif") + "-condensates_AveProj.tif"
    imwrite(fsave_left, img_left_aveproj.astype("int16"), imagej=True)

    # crop, discard last 10 frames
    img_right_cropped = crop_imgstack(img_right_aligned)
    img_right_final = img_right_cropped[:-10, :, :]
    fsave_right = fpath.strip(".tif") + "-RNAs.tif"
    imwrite(
        fsave_right,
        img_right_final,
        imagej=True,
        metadata={"axes": "TYX"},
    )
