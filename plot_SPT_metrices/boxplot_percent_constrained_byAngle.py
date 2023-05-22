from os.path import join, dirname, basename
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="white")

pd.options.mode.chained_assignment = None  # default='warn'

color_palette = [
    "#001219",
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#e9d8a6",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
    "#ae2012",
    "#9b2226",
]
os.chdir(
    "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures/May2023_wrapup"
)

dict_input_path = {
    "0Dex, -, 0h": "Angles_Mobile_tracks-0Dex_noTotR_0h.csv",
    "0Dex, -, 3h": "Angles_Mobile_tracks-0Dex_noTotR_3h.csv",
    "0Dex, He, 0h": "Angles_Mobile_tracks-0Dex_Hela_0h.csv",
    "0Dex, Ce, 0h": "Angles_Mobile_tracks-0Dex_Cerebral_0h.csv",
    "0Dex, Sp, 0h": "Angles_Mobile_tracks-0Dex_Spinal_0h.csv",
    "10Dex, -, 0h": "Angles_Mobile_tracks-10Dex_noTotR_0h.csv",
    "10Dex, -, 3h": "Angles_Mobile_tracks-10Dex_noTotR_3h.csv",
    "10Dex, He, 0h": "Angles_Mobile_tracks-10Dex_Hela_0h.csv",
    "10Dex, Ce, 0h": "Angles_Mobile_tracks-10Dex_Cerebral_0h.csv",
    "10Dex, Sp, 0h": "Angles_Mobile_tracks-10Dex_Spinal_0h.csv",
}

lst_tag = []
lst_FOVname = []
lst_N_mobile = []
lst_N_constrained = []
lst_fraction_constrained = []
for key in dict_input_path.keys():
    df_current_file = pd.read_csv(dict_input_path[key])

    lst_keys = df_current_file.keys().tolist()
    N_bins = len(lst_keys) - 5
    threshold_last_bin_probability = (1 / N_bins) * 1.5

    df_current_file = df_current_file.astype({lst_keys[-1]: float})

    for FOVname in df_current_file.filename.unique():
        df_currentFOV = df_current_file[df_current_file.filename == FOVname]

        N_mobile = df_currentFOV.shape[0]
        N_constrained = np.sum(
            df_currentFOV[lst_keys[-1]].to_numpy() > threshold_last_bin_probability
        )

        fraction_constrained = N_constrained / N_mobile

        lst_tag.append(key)
        lst_FOVname.append(FOVname)
        lst_N_mobile.append(N_mobile)
        lst_N_constrained.append(N_constrained)
        lst_fraction_constrained.append(fraction_constrained)


df_save = pd.DataFrame(
    {
        "label": lst_tag,
        "FOVname": lst_FOVname,
        "N, Mobile": lst_N_mobile,
        "N, Constrained, by Angle": lst_N_constrained,
        "Constrained Fraction, by Angle": lst_fraction_constrained,
    },
    dtype=object,
)
df_save.to_csv("N_and_Fraction_per_FOV_byAngle.csv", index=False)


plt.figure(figsize=(8, 5), dpi=300)
ax = sns.boxplot(
    data=df_save,
    x="label",
    y="Constrained Fraction, by Angle",
    palette=color_palette,
)
ax = sns.stripplot(
    data=df_save,
    x="label",
    y="Constrained Fraction, by Angle",
    color="0.4",
)
plt.title("Constrained Fraction per FOV, by Angle", weight="bold")
plt.ylabel("Constrained Fraction, by Angle", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig("boxplot_constrained_fraction_byAngle.png", format="png")
plt.close()
