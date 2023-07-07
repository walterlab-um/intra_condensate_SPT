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

# Displacement threshold for non static molecules
threshold_disp = 0.2  # unit: um
# alpha component threshold for constrained diffusion
threshold_alpha = 0.5

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
# calculate error bounds
s_per_frame = 0.02
loc_err = 0.03
log10D_low = np.log10(loc_err**2 / (4 * (s_per_frame)))
# log10D_low = -3

# Output file structure
columns = [
    "label",
    "replicate_prefix",
    "N, Total",
    "N, Mobile",
    "N, Constrained",
    "Static Fraction",
    "Constrained Fraction",
]
# construct output dataframe
lst_rows_of_df = []
for key in track(dict_input_path.keys()):
    df_current = pd.read_csv(dict_input_path[key])
    df_current = df_current.astype(
        {"linear_fit_log10D": float, "Displacement_um": float, "alpha": float}
    )
    # all filenames within the current condition/file
    all_filenames = df_current["filename"].unique().tolist()
    # filename prefix for each replicate
    replicate_prefixs = np.unique([f.split("FOV")[0] for f in all_filenames])

    for prefix in replicate_prefixs:
        current_replicate_filenames = [f for f in all_filenames if prefix in f]
        df_current_replicate = df_current[
            df_current["filename"].isin(current_replicate_filenames)
        ]

        df_mobile_byD = df_current_replicate[
            df_current_replicate["linear_fit_log10D"] > log10D_low
        ]
        df_mobile = df_mobile_byD[df_mobile_byD["Displacement_um"] >= threshold_disp]
        df_constrained = df_mobile[df_mobile["alpha"] <= threshold_alpha]

        N_total = df_current_replicate.shape[0]
        N_mobile = df_mobile.shape[0]
        N_constrained = df_constrained.shape[0]

        if N_constrained < 1:
            continue

        F_static = (N_total - N_mobile) / N_total
        F_constrained = N_constrained / N_mobile

        # save
        lst_rows_of_df.append(
            [
                key,
                prefix,
                N_total,
                N_mobile,
                N_constrained,
                F_static,
                F_constrained,
            ]
        )

df_save = pd.DataFrame.from_records(
    lst_rows_of_df,
    columns=columns,
)
df_save.to_csv("N_and_Fraction_per_replicate.csv", index=False)

df_plot = df_save[df_save["N, Total"] > 1000].melt(
    id_vars=["label"],
    value_vars=["Static Fraction", "Constrained Fraction"],
)
