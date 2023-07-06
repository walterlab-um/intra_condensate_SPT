import os
import pandas as pd
from rich.progress import track

folder_data = "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig2_diffusion analysis/SPT_results_AIO_files"

# concat all csv files, perform this
os.chdir(folder_data)
lst_df = []
for f in track(os.listdir(folder_data)):
    if f.startswith("SPT_results_AIO"):
        lst_df.append(pd.read_csv(f))
df_all = pd.concat(lst_df)

df_all.to_csv("SPT_results_AIO_concat-pleaserename.csv", index=False)
