from tkinter import filedialog as fd
import os
from os.path import dirname, basename
import pandas as pd
from random import sample
from rich.progress import track


fpath = fd.askopenfilename()

df = pd.read_csv(fpath)
os.chdir(dirname(fpath))

for i in track(range(5)):
    indexes_picked = sample(range(df.shape[0], k=round(df.shape[0] / 10)))
    df_out = df.iloc[indexes_picked]
    df_out.to_csv(basename(fpath)[:-4] + "-sample-" + str(i) + ".csv", index=False)
    i += 1
