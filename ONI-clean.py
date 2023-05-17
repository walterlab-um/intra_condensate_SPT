import os
from os.path import join, isfile
import shutil
from tkinter import filedialog as fd
from rich.progress import track
from rich import print as rprint


# To install rich for the progress bar, run the following in command line:
# python -m pip install rich

rprint("[red]Choose an experiment folder[red]")
folderpath = fd.askdirectory(initialdir="/Volumes/AnalysisGG/RAW_DATA_ORGANIZED")
# /Volumes/nwalter-group/Guoming Gao/RAW_DATA/DEFAULT_USER
# /Volumes/AnalysisGG/RAW_DATA_ORGANIZED


print("Now organizing:\n", folderpath)

try:
    os.mkdir(join(folderpath, "non-tif"))
except:
    pass


# leaving out files ending with .tif and .txt for experiment notes
lst_nontif = [
    f
    for f in os.listdir(folderpath)
    if isfile(join(folderpath, f))
    & (
        (not f.endswith(".tif"))
        and (not f.endswith(".txt"))
        and (not f.endswith(".tiff"))
    )
]

for f in track(lst_nontif, description="Processing..."):
    shutil.move(join(folderpath, f), join(folderpath, "non-tif", f))
