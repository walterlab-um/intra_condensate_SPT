import os
from os.path import dirname, join
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import seaborn as sns

sns.set(color_codes=True, style="white")

# plot saSPT results for all replicates in a single condition
# Must pool together, because the dataset size only fullfill saSPT's requirements when it's pooled.

os.chdir(
    "/Volumes/lsa-nwalter/Guoming_Gao_turbo/Walterlab_server/PROCESSED_DATA/RNA-diffusion-in-FUS/RNAinFUS_PaperFigures/Fig3_aging crowding Hela"
)

lst_fname = [
    "saSPT-pooled-0Dex_noTR_0hr.csv",
    "saSPT-pooled-0Dex_noTR_3hr.csv",
    "saSPT-pooled-0Dex_noTR_6hr.csv",
    "saSPT-pooled-0Dex_noTR_8hr.csv",
    "saSPT-pooled-0Dex_helaTR_1hr.csv",
    "saSPT-pooled-10Dex_noTR_0hr.csv",
    "saSPT-pooled-10Dex_noTR_3hr.csv",
    "saSPT-pooled-10Dex_noTR_6hr.csv",
    "saSPT-pooled-10Dex_noTR_8hr.csv",
    "saSPT-pooled-10Dex_helaTR_0hr.csv",
]

# for D distribution
plt_ylim = (0, 0.015)
# for SA heat map
cut_off_quantile = 0.99
# for fractin
static_threshold = -1.9

color_palette = [
    "#9b2226",
    "#8d2a2e",
    "#582326",
    "#333232",
]

color_palette_2 = [
    "#9b2226",
    "#f7b801",
]


def extract_log10D_density(df_current_file):
    range_D = df_current_file["diff_coef"].unique()
    log10D_density = []
    for log10D in range_D:
        df_current_log10D = df_current_file[df_current_file["diff_coef"] == log10D]
        log10D_density.append(df_current_log10D["mean_posterior_occupation"].sum())

    df_toplot = pd.DataFrame(
        {"log10D": np.log10(range_D), "Probability": log10D_density}, dtype=float
    )

    return df_toplot


lst_frac_static = []
for fname_data in lst_fname:
    df_saSPT = pd.read_csv(fname_data, dtype=float)
    df_toplot = extract_log10D_density(df_saSPT)
    log10D = df_toplot["log10D"].to_numpy(dtype=float)
    proportion = df_toplot["Probability"].to_numpy(dtype=float)
    F_static = proportion[log10D < static_threshold].sum()
    lst_frac_static.append(F_static)

#############################################
plt.figure(figsize=(5, 3), dpi=300)
i = 0
for fname_data in lst_fname[0:4]:
    df_saSPT = pd.read_csv(fname_data, dtype=float)
    df_toplot = extract_log10D_density(df_saSPT)
    sns.lineplot(
        data=df_toplot,
        x="log10D",
        y="Probability",
        color=color_palette[i],
        label=fname_data.split("TR_")[-1].split("hr")[0] + " hr",
        alpha=0.7,
    )
    # find peaks
    log10D = df_toplot["log10D"].to_numpy(dtype=float)
    proportion = df_toplot["Probability"].to_numpy(dtype=float)
    # only find peaks that are separate more than delta_log10D > 0.5
    # peaks_idx, _ = find_peaks(proportion, distance=int(0.5 / (log10D[1] - log10D[0])))
    peaks_idx, _ = find_peaks(proportion)
    for x in log10D[peaks_idx]:
        plt.axvline(x, color=color_palette[i], ls="--", lw=1, alpha=0.3)
    i += 1
plt.title("Aging, no Dextran", weight="bold")
plt.xlim(log10D.min(), log10D.max())
plt.ylim(plt_ylim[0], plt_ylim[1])
plt.xlabel(r"Apparent log$_{10}$D, $\mu$m$^2$/s", weight="bold")
plt.ylabel("SA Occupation", weight="bold")
plt.tight_layout()
plt.savefig(
    "saSPT_pooled-Aging_noDex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()

# bar plot of F_static
plt.figure(figsize=(3, 5), dpi=300)
plt.bar(
    x=["0 h", "3 h", "6 h", "8 h"],
    height=lst_frac_static[0:4],
    color=color_palette,
)
plt.ylim(0, 1)
plt.title("Aging, without Dextran", weight="bold")
plt.ylabel("Static Fraction", weight="bold")
plt.tight_layout()
plt.savefig(
    "F_static-saSPT_pooled-Aging_noDex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()

#############################################
plt.figure(figsize=(5, 3), dpi=300)
i = 0
for fname_data in lst_fname[5:-1]:
    df_saSPT = pd.read_csv(fname_data, dtype=float)
    df_toplot = extract_log10D_density(df_saSPT)
    sns.lineplot(
        data=df_toplot,
        x="log10D",
        y="Probability",
        color=color_palette[i],
        label=fname_data.split("TR_")[-1].split("hr")[0] + " hr",
        alpha=0.7,
    )
    # find peaks
    log10D = df_toplot["log10D"].to_numpy(dtype=float)
    proportion = df_toplot["Probability"].to_numpy(dtype=float)
    # only find peaks that are separate more than delta_log10D > 0.5
    # peaks_idx, _ = find_peaks(proportion, distance=int(0.5 / (log10D[1] - log10D[0])))
    peaks_idx, _ = find_peaks(proportion)
    for x in log10D[peaks_idx]:
        plt.axvline(x, color=color_palette[i], ls="--", lw=1, alpha=0.3)
    i += 1
plt.title("Aging, 10% Dextran", weight="bold")
plt.xlim(log10D.min(), log10D.max())
plt.ylim(plt_ylim[0], plt_ylim[1])
plt.xlabel(r"Apparent log$_{10}$D, $\mu$m$^2$/s", weight="bold")
plt.ylabel("SA Occupation", weight="bold")
plt.tight_layout()
plt.savefig(
    "saSPT_pooled-Aging_10Dex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()

# bar plot of F_static
plt.figure(figsize=(3, 5), dpi=300)
plt.bar(
    x=["0 h", "3 h", "6 h", "8 h"],
    height=lst_frac_static[5:-1],
    color=color_palette,
)
plt.ylim(0, 1)
plt.title("Aging, 10% Dextran", weight="bold")
plt.ylabel("Static Fraction", weight="bold")
plt.tight_layout()
plt.savefig(
    "F_static-saSPT_pooled-Aging_10Dex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()

#############################################
plt.figure(figsize=(5, 3), dpi=300)
i = 0
lst_fname_RNA = [lst_fname[0], lst_fname[4]]
lst_label = ["-RNA", "+RNA"]
for fname_data in lst_fname_RNA:
    df_saSPT = pd.read_csv(fname_data, dtype=float)
    df_toplot = extract_log10D_density(df_saSPT)
    sns.lineplot(
        data=df_toplot,
        x="log10D",
        y="Probability",
        color=color_palette_2[i],
        label=lst_label[i],
        alpha=0.7,
    )
    # find peaks
    log10D = df_toplot["log10D"].to_numpy(dtype=float)
    proportion = df_toplot["Probability"].to_numpy(dtype=float)
    # only find peaks that are separate more than delta_log10D > 0.5
    # peaks_idx, _ = find_peaks(proportion, distance=int(0.5 / (log10D[1] - log10D[0])))
    peaks_idx, _ = find_peaks(proportion)
    for x in log10D[peaks_idx]:
        plt.axvline(x, color=color_palette_2[i], ls="--", lw=1, alpha=0.3)
    i += 1
plt.title("Effect of RNA, no Dextran", weight="bold")
plt.xlim(log10D.min(), log10D.max())
plt.ylim(plt_ylim[0], plt_ylim[1])
plt.xlabel(r"Apparent log$_{10}$D, $\mu$m$^2$/s", weight="bold")
plt.ylabel("SA Occupation", weight="bold")
plt.tight_layout()
plt.savefig(
    "saSPT_pooled-Aging_compareRNA_noDex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()

# bar plot of F_static
plt.figure(figsize=(2, 5), dpi=300)
plt.bar(
    x=lst_label,
    height=[lst_frac_static[0], lst_frac_static[4]],
    color=color_palette_2,
    width=0.8,
)
plt.ylim(0, 1)
plt.title("Effect of RNA,\nno Dextran", weight="bold")
plt.ylabel("Static Fraction", weight="bold")
plt.tight_layout()
plt.savefig(
    "F_static-saSPT_pooled-Aging_compareRNA_noDex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()

#############################################
plt.figure(figsize=(5, 3), dpi=300)
i = 0
lst_fname_RNA = [lst_fname[5], lst_fname[-1]]
lst_label = ["-RNA", "+RNA"]
for fname_data in lst_fname_RNA:
    df_saSPT = pd.read_csv(fname_data, dtype=float)
    df_toplot = extract_log10D_density(df_saSPT)
    sns.lineplot(
        data=df_toplot,
        x="log10D",
        y="Probability",
        color=color_palette_2[i],
        label=lst_label[i],
        alpha=0.7,
    )
    # find peaks
    log10D = df_toplot["log10D"].to_numpy(dtype=float)
    proportion = df_toplot["Probability"].to_numpy(dtype=float)
    # only find peaks that are separate more than delta_log10D > 0.5
    # peaks_idx, _ = find_peaks(proportion, distance=int(0.5 / (log10D[1] - log10D[0])))
    peaks_idx, _ = find_peaks(proportion)
    for x in log10D[peaks_idx]:
        plt.axvline(x, color=color_palette_2[i], ls="--", lw=1, alpha=0.3)
    i += 1
plt.title("Effect of RNA, 10% Dextran", weight="bold")
plt.xlim(log10D.min(), log10D.max())
plt.ylim(plt_ylim[0], plt_ylim[1])
plt.xlabel(r"Apparent log$_{10}$D, $\mu$m$^2$/s", weight="bold")
plt.ylabel("SA Occupation", weight="bold")
plt.tight_layout()
plt.savefig(
    "saSPT_pooled-Aging_compareRNA_10Dex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()

# bar plot of F_static
plt.figure(figsize=(2, 5), dpi=300)
plt.bar(
    x=lst_label,
    height=[lst_frac_static[5], lst_frac_static[-1]],
    color=color_palette_2,
    width=0.8,
)
plt.ylim(0, 1)
plt.title("Effect of RNA,\n10% Dextran", weight="bold")
plt.ylabel("Static Fraction", weight="bold")
plt.tight_layout()
plt.savefig(
    "F_static-saSPT_pooled-Aging_compareRNA_10Dex.png",
    format="png",
    bbox_inches="tight",
)
plt.close()


# ## plot SA heatmap
# all_diff_coef = df_saSPT["diff_coef"].unique()
# all_loc_error = df_saSPT["loc_error"].unique()
# N_diff_coef = all_diff_coef.shape[0]
# N_loc_error = all_loc_error.shape[0]
# heatmap = np.zeros((N_loc_error, N_diff_coef))
# for row in np.arange(N_loc_error):
#     loc_error = all_loc_error[row]
#     df_current_row = df_saSPT[df_saSPT["loc_error"] == loc_error]
#     for column in np.arange(N_diff_coef):
#         diff_coef = all_diff_coef[column]
#         current_cell = df_current_row[df_current_row["diff_coef"] == diff_coef]
#         heatmap[row, column] = current_cell["mean_posterior_occupation"]

# df_heatmap = pd.DataFrame(
#     heatmap,
#     columns=np.around(np.log10(all_diff_coef), 1),
#     index=np.rint(all_loc_error * 1000),
# )

# plt.figure(dpi=300, figsize=(7, 4))
# quantile = np.quantile(
#     df_saSPT["mean_posterior_occupation"], cut_off_quantile
# )  # add cut off
# ax = sns.heatmap(
#     data=df_heatmap,
#     cmap="Reds",
#     norm=LogNorm(
#         vmin=df_saSPT["mean_posterior_occupation"].min(),
#         vmax=quantile,
#     ),
#     xticklabels=10,
#     yticklabels=7,
#     cbar=True,
#     cbar_kws={"orientation": "horizontal"},
# )
# ax.invert_yaxis()
# ax.tick_params(axis="both", which="major", labelsize=11)
# plt.xlabel(r"Apparent log$_{10}$D, $\mu$m$^2$/s", weight="bold")
# plt.ylabel("Localization Error, nm", weight="bold")
# plt.axis("scaled")
# plt.tight_layout()
# plt.savefig(
#     join(dirname(fname_data), "saSPT_pooled-0Dex_noTR_0hr-SA.png"),
#     format="png",
#     bbox_inches="tight",
# )
# plt.close()