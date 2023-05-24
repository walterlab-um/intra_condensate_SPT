import os
from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from rich.progress import track

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
folder_save = "Hypothesis_two_population_short_constrained_but_not_long"

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
lst_keys = list(dict_input_path.keys())
for i in track(range(len(lst_keys))):
    key = lst_keys[i]
    df_data = pd.read_csv(dict_input_path[key])
    color = color_palette[i]

    last_column_name = df_data.keys().tolist()[-1]
    df_data = df_data.astype(
        {last_column_name: float, "N_steps": float, "Displacement_um": float}
    )
    df_data = df_data[df_data["Displacement_um"] > 0.2]

    plt.figure(figsize=(5, 5), dpi=300)
    sns.jointplot(
        data=df_data,
        x=last_column_name,
        y="N_steps",
        kind="kde",
        color=color,
        fill=True,
        thresh=0,
        levels=100,
        cut=0,
        clip=((0, 0.5), (8, 100)),
        norm=LogNorm(),
    )
    title = " ".join(dict_input_path[key][:-4].split("_"))
    plt.title(title, weight="bold")
    plt.ylabel("Number of Steps in a Trajectory", weight="bold")
    plt.xlabel("Probability of Angle within " + last_column_name, weight="bold")
    plt.ylim(8, 100)
    plt.xlim(0, 0.5)
    plt.tight_layout()
    plt.savefig(
        join(folder_save, "N_vs_constrained_" + dict_input_path[key][:-4] + ".png"),
        format="png",
    )
    plt.close()

    # plt.figure(figsize=(6, 6), dpi=300)
    # sns.jointplot(
    #     data=df_data,
    #     x=last_column_name,
    #     y="Displacement_um",
    #     kind="kde",
    #     color=color,
    #     fill=True,
    #     thresh=0,
    #     levels=100,
    #     cut=0,
    #     clip=((0, 0.5), (0.2, 0.5)),
    #     norm=LogNorm(),
    # )
    # title = " ".join(dict_input_path[key][:-4].split("_"))
    # plt.title(title, weight="bold")
    # plt.ylabel(r"Track Displacement, $\mu$m", weight="bold")
    # plt.xlabel("Probability of Angle within " + last_column_name, weight="bold")
    # plt.ylim(0.2, 0.5)
    # plt.xlim(0, 0.5)
    # plt.tight_layout()
    # plt.savefig(
    #     join(folder_save, "Disp_vs_constrained_" + dict_input_path[key][:-4] + ".png"),
    #     format="png",
    # )
    # plt.close()
