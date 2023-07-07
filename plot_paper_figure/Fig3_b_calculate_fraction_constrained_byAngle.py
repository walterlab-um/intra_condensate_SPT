import os
import numpy as np
import pandas as pd
import seaborn as sns
from rich.progress import track

sns.set(color_codes=True, style="white")
pd.options.mode.chained_assignment = None  # default='warn'

os.chdir(
    "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_aging crowding Hela"
)
# threshold 1: minimla tracklength, because too short a track is too stochastic in angle distribution
threshold_tracklength = 50
# threshold 2: probability of an angle falls into the last bin, need a factor like 2 here to account for the stochastic nature of SM track
N_bins = 6
threshold_last_bin_probability = (1 / N_bins) * 2
# threshold 3: D error bounds to determine static molecule
s_per_frame = 0.02
static_err = 0.016
threshold_log10D = np.log10(static_err**2 / (4 * (s_per_frame)))
# threshold 4: Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um

dict_input_path = {
    "0Dex, -, 0h": "SPT_results_AIO_concat-0Dex_noTR_0hr.csv",
    "0Dex, -, 3h": "SPT_results_AIO_concat-0Dex_noTR_3hr.csv",
    "0Dex, -, 6h": "SPT_results_AIO_concat-0Dex_noTR_6hr.csv",
    "0Dex, -, 8h": "SPT_results_AIO_concat-0Dex_noTR_8hr.csv",
    "0Dex, Hela, 0h": "SPT_results_AIO_concat-0Dex_helaTR_1hr.csv",
    "10Dex, -, 0h": "SPT_results_AIO_concat-10Dex_noTR_0hr.csv",
    "10Dex, -, 3h": "SPT_results_AIO_concat-10Dex_noTR_3hr.csv",
    "10Dex, -, 6h": "SPT_results_AIO_concat-10Dex_noTR_6hr.csv",
    "10Dex, -, 8h": "SPT_results_AIO_concat-10Dex_noTR_8hr.csv",
    "10Dex, Hela, 0h": "SPT_results_AIO_concat-10Dex_helaTR_0hr.csv",
}

# Output file structure
columns = [
    "label",
    "replicate_prefix",
    "N, Mobile",
    "N, Constrained",
    "Constrained Fraction, by Angle",
]
# construct output dataframe
lst_rows_of_df = []
for key in track(dict_input_path.keys()):
    df_current_file = pd.read_csv(dict_input_path[key])
    lst_keys = df_current_file.keys().tolist()
    df_current_file = df_current_file.astype(
        {
            "linear_fit_log10D": float,
            "N_steps": int,
            "Displacement_um": float,
            lst_keys[-1]: float,
        }
    )
    # filter out static tracks
    df_longtracks = df_current_file[df_current_file["N_steps"] >= threshold_tracklength]
    df_mobile_byD = df_longtracks[
        df_longtracks["linear_fit_log10D"] >= threshold_log10D
    ]
    df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]
    # all filenames within the current condition/file
    all_filenames = df_mobile["filename"].unique().tolist()
    # filename prefix for each replicate
    replicate_prefixs = np.unique([f.split("FOV")[0] for f in all_filenames])

    for prefix in replicate_prefixs:
        current_replicate_filenames = [f for f in all_filenames if prefix in f]
        df_current_replicate = df_mobile[
            df_mobile["filename"].isin(current_replicate_filenames)
        ]

        N_mobile = df_current_replicate.shape[0]
        N_constrained = np.sum(
            df_current_replicate[lst_keys[-1]].to_numpy()
            > threshold_last_bin_probability
        )

        fraction_constrained = N_constrained / N_mobile

        # save
        lst_rows_of_df.append(
            [
                key,
                prefix,
                N_mobile,
                N_constrained,
                fraction_constrained,
            ]
        )


df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
df_save.to_csv("N_and_Fraction_per_replicate_byAngle.csv", index=False)
