from os.path import join, dirname, basename
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True, style="white")

pd.options.mode.chained_assignment = None  # default='warn'

color_palette = [
    "#ffcb05",
    "#b79915",
    "#c8be56",
    "#546a4e",
    "#3b6f7d",
    "#00274c",
    "#656ba4",
    "#4d2d38",
    "#c5706d",
    "#9a3324",
]
os.chdir("/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/")
path_save = "/Volumes/nwalter-group/Guoming Gao/PROCESSED_DATA/RNA-diffusion-in-FUS/PaperFigures"
dict_input_path = {
    "0Dex, -, 0h": "bioFUStether-10FUS-1Mg-noDex-RT/No Total RNA/20221031_0hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "0Dex, -, 3h": "bioFUStether-10FUS-1Mg-noDex-RT/No Total RNA/20221031_3hrs/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "0Dex, He, 1h": "bioFUStether-10FUS-1Mg-noDex-RT/Total RNA Background/HeLa Total RNA/20221031_1hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "0Dex, Ce, 1h": "bioFUStether-10FUS-1Mg-noDex-RT/Total RNA Background/Cerebral Total RNA/20221114_1hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "0Dex, Sp, 1h": "bioFUStether-10FUS-1Mg-noDex-RT/Total RNA Background/Spinal Total RNA/20221031_1hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "10Dex, -, 0h": "bioFUStether-10FUS-1Mg-10Dex-RT/NoTotalRNA/FL-0hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "10Dex, -, 3h": "bioFUStether-10FUS-1Mg-10Dex-RT/NoTotalRNA/FL-3hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "10Dex, He, 1h": "bioFUStether-10FUS-1Mg-10Dex-RT/Total RNA Background/HelaTotalRNA/FL-0hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "10Dex, Ce, 1h": "bioFUStether-10FUS-1Mg-10Dex-RT/Total RNA Background/Cerebral TotalRNA/FL-0hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
    "10Dex, Sp, 1h": "bioFUStether-10FUS-1Mg-10Dex-RT/Total RNA Background/SpinalTotalRNA/FL-0hr/MSDtau-D-alpha/EffectiveD-alpha-alltracks.csv",
}

lst_total_size = []
lst_N_linear = []
lst_N_loglog = []
for key in dict_input_path.keys():
    df_current = pd.read_csv(dict_input_path[key], dtype=float)
    lst_total_size.append(df_current.shape[0])
    lst_N_linear.append(df_current[df_current["slope_linear"] > 0].shape[0])
    lst_N_loglog.append(df_current[df_current["alpha"] > 0].shape[0])

df_plot = pd.DataFrame(
    {
        "label": dict_input_path.keys(),
        "N_total": lst_total_size,
        "N_linear": lst_N_linear,
        "N_loglog": lst_N_loglog,
    },
    dtype=object,
)

plt.figure(figsize=(7, 5), dpi=200)
ax = sns.barplot(
    data=df_plot,
    x="label",
    y="N_total",
    width=0.9,
    palette=color_palette,
)
plt.title("Dataset Size", weight="bold")
plt.ylabel("Total # of trajectories", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig(join(path_save, "barplot_dataset_size_total.png"), format="png")
plt.close()

plt.figure(figsize=(7, 5), dpi=200)
ax = sns.barplot(
    data=df_plot,
    x="label",
    y="N_linear",
    width=0.9,
    palette=color_palette,
)
plt.title("Dataset Size", weight="bold")
plt.ylabel("Total # of trajectories in linear fitting", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig(join(path_save, "barplot_dataset_size_linear.png"), format="png")
plt.close()

plt.figure(figsize=(7, 5), dpi=200)
ax = sns.barplot(
    data=df_plot,
    x="label",
    y="N_loglog",
    width=0.9,
    palette=color_palette,
)
plt.title("Dataset Size", weight="bold")
plt.ylabel("Total # of trajectories in log-log fitting", weight="bold")
ax.xaxis.set_tick_params(labelsize=15, labelrotation=90)
plt.xlabel("")
plt.tight_layout()
plt.savefig(join(path_save, "barplot_dataset_size_loglog.png"), format="png")
plt.close()
