import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.progress import track
from scipy.optimize import curve_fit

sns.set(color_codes=True, style="white")

folderpath = "/Volumes/AnalysisGG/PROCESSED_DATA/RNA-diffusion-in-FUS/20220712_FLmRNA_10FUS_1Mg_10Dex_noTotR_24C/"
os.chdir(folderpath)
lst_fname = [
    f
    for f in os.listdir(folderpath)
    if f.endswith("RNA_linregress_D.csv") & f.startswith("high")
]
R2threshold = 0.9
D_threshold = -0.8


def pool_R2filter(lst_fname, folderpath, R2threshold):
    lst_log10D = []
    lst_R = []
    lst_mindistance = []
    lst_distnorm = []
    for fname in lst_fname:
        df_in = pd.read_csv(join(folderpath, fname))
        # R^2 filtering
        df_R2above = df_in[df_in["R2"] >= R2threshold]
        lst_log10D.extend(list(df_R2above["log10D"]))
        lst_R.extend(list(df_R2above["condensate R (nm)"]))
        lst_mindistance.extend(list(df_R2above["min distance (nm)"]))
        lst_distnorm.extend(
            list(df_R2above["min distance (nm)"] / df_R2above["condensate R (nm)"])
        )

    df_plot = pd.DataFrame(
        {
            "log10D (um^2/s)": lst_log10D,
            "condensate R (nm)": lst_R,
            "distance to edge (nm)": lst_mindistance,
            "normalized distance": lst_distnorm,
        },
        dtype=float,
    )
    return df_plot


def DualGauss_fit_plot_text(data):
    counts, bins = np.histogram(data, bins=30)
    # Define the dual peak and single peak Gaussian function
    def DualGauss(x, A1, x1, sigma1, A2, x2, sigma2):
        return A1 * np.exp((-1 / 2) * ((x - x1) ** 2 / sigma1 ** 2)) + A2 * np.exp(
            (-1 / 2) * ((x - x2) ** 2 / sigma2 ** 2)
        )

    def Gauss(x, A, x0, sigma):
        return A * np.exp((-1 / 2) * ((x - x0) ** 2 / sigma ** 2))

    # Fit to DualGauss and plot individually and combined
    (A1, x1, sigma1, A2, x2, sigma2), pcov = curve_fit(
        DualGauss, (bins[1:] + bins[:-1]) / 2, counts
    )
    err_A1, err_x1, err_sigma1, err_A2, err_x2, err_sigma2 = np.sqrt(np.diag(pcov))
    curve_x = np.arange(bins[0], bins[-1], 0.01)
    curve_ydual = DualGauss(curve_x, A1, x1, sigma1, A2, x2, sigma2)
    curve_y1 = Gauss(curve_x, A1, x1, sigma1)
    curve_y2 = Gauss(curve_x, A2, x2, sigma2)
    plt.plot(curve_x, curve_ydual, color="dimgray", linewidth=2)
    plt.plot(curve_x, curve_y1, color=sns.color_palette()[1], linewidth=2)
    plt.plot(curve_x, curve_y2, color=sns.color_palette()[3], linewidth=2)
    plt.axvline(x=x1, color=sns.color_palette()[1], ls="--")
    plt.axvline(x=x2, color=sns.color_palette()[3], ls="--")

    # label with text
    plt.text(
        0.76,
        0.8,
        "log10D$_1$ = " + str(round(x1, 2)) + "$\pm$" + str(round(err_x1, 2)),
        weight="bold",
        fontsize=11,
        color=sns.color_palette()[1],
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.76,
        0.75,
        "log10D$_2$ = " + str(round(x2, 2)) + "$\pm$" + str(round(err_x2, 2)),
        weight="bold",
        fontsize=11,
        color=sns.color_palette()[3],
        transform=plt.gcf().transFigure,
    )


def Gauss_fit_plot_text(data, coloridx, text_x, text_y):
    counts, bins = np.histogram(data, bins=30)

    def Gauss(x, A, x0, sigma):
        return A * np.exp((-1 / 2) * ((x - x0) ** 2 / sigma ** 2))

    # Fit to Gauss and plot individually and combined
    (A, x0, sigma), pcov = curve_fit(
        Gauss,
        (bins[1:] + bins[:-1]) / 2,
        counts,
        method="dogbox",
        bounds=([0, 0, 0], [0.5 * len(data), 2500, 1000]),
    )
    err_A, err_x0, err_sigma = np.sqrt(np.diag(pcov))
    curve_x = np.arange(bins[0], bins[-1], 0.01)
    curve_y = Gauss(curve_x, A / len(data), x0, sigma)
    plt.plot(curve_x, curve_y, color=sns.color_palette()[coloridx], linewidth=2)
    plt.axvline(x=x0, color=sns.color_palette()[coloridx], ls="--")

    # label with text
    plt.text(
        text_x,
        text_y,
        "$\mu$ = " + str(round(x0, 2)) + "$\pm$" + str(round(err_x0, 2)),
        weight="bold",
        fontsize=11,
        color=sns.color_palette()[coloridx],
        transform=plt.gcf().transFigure,
    )


def axvline_bounds(s_per_frame, max_link_pxl):
    # calculate limits
    # lower bounds determiend by static localization error 55 nm
    low = np.log10(0.055 ** 2 / (4 * s_per_frame))
    # higher bounds determiend by max linking length 3 pixels
    high = np.log10((0.117 * max_link_pxl) ** 2 / (4 * s_per_frame))
    # plot limits
    plt.axvline(x=low, color="dimgray", ls=":", alpha=0.5)
    plt.axvline(x=high, color="dimgray", ls=":", alpha=0.5)


######################################
# Prepare DataFrame, filter by fitting R2, with D spliting
df_plot = pool_R2filter(lst_fname, folderpath, R2threshold)
# D splitting
df_plot = df_plot.sort_values(by=["log10D (um^2/s)"])
N_slow = df_plot[df_plot["log10D (um^2/s)"] < D_threshold].shape[0]
N_fast = df_plot.shape[0] - N_slow
tag_slow = "log10D<" + str(D_threshold) + ", N=" + str(N_slow)
tag_fast = "log10D>" + str(D_threshold) + ", N=" + str(N_fast)
new_column = list(np.repeat(tag_slow, N_slow)) + list(np.repeat(tag_fast, N_fast))
df_plot["tag"] = new_column


######################################
# D-Condensate Size KDE
plt.figure(figsize=(5, 5), dpi=200)
sns.kdeplot(
    data=df_plot,
    x="log10D (um^2/s)",
    y="condensate R (nm)",
    alpha=0.7,
    fill=True,
    legend=False,
)
plt.text(
    0.8,
    0.88,
    "N = " + str(df_plot.shape[0]),
    fontsize=11,
    transform=plt.gcf().transFigure,
    weight="bold",
)
plt.title("Diffusion Coefficient-Condensate Size", fontsize=13, fontweight="bold")
plt.xlabel("log10D ($\mu$m^2/s)", weight="bold")
plt.ylabel("Condensate's Equivalent Radius (nm)")
plt.tight_layout()
fsave = "D-R-kde.png"
plt.savefig(fsave, format="png")


######################################
# D-Distance to Edge KDE
plt.figure(figsize=(5, 5), dpi=200)
sns.kdeplot(
    data=df_plot,
    x="log10D (um^2/s)",
    y="distance to edge (nm)",
    alpha=0.7,
    fill=True,
    legend=False,
)
plt.text(
    0.8,
    0.88,
    "N = " + str(df_plot.shape[0]),
    fontsize=11,
    transform=plt.gcf().transFigure,
    weight="bold",
)
plt.title("Diffusion Coefficient-Distance to Edge", fontsize=13, fontweight="bold")
plt.xlabel("log10D ($\mu$m^2/s)", weight="bold")
plt.ylabel("Distance to Edge (nm)")
# plt.legend()
plt.tight_layout()
fsave = "D-distance-kde.png"
plt.savefig(fsave, format="png")


######################################
# D-Normalized Distance to Edge KDE
plt.figure(figsize=(5, 5), dpi=200)
sns.kdeplot(
    data=df_plot,
    x="log10D (um^2/s)",
    y="normalized distance",
    alpha=0.7,
    fill=True,
    legend=False,
)
plt.text(
    0.8,
    0.845,
    "N = " + str(df_plot.shape[0]),
    fontsize=11,
    transform=plt.gcf().transFigure,
    weight="bold",
)
plt.title(
    "Diffusion Coefficient-\nNormalized Distance to Edge",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("log10D ($\mu$m^2/s)", weight="bold")
plt.ylabel("Normalized Distance to Edge")
# plt.legend()
plt.tight_layout()
fsave = "D-normalized-distance-kde.png"
plt.savefig(fsave, format="png")


######################################
# Total D distribution with fitting
plt.figure(figsize=(9, 4), dpi=200)
g = sns.histplot(
    data=df_plot,
    x="log10D (um^2/s)",
    fill=True,
    stat="count",
    alpha=0.3,
    color="dimgray",
    bins=30,
)
DualGauss_fit_plot_text(df_plot["log10D (um^2/s)"])
axvline_bounds(s_per_frame=0.02, max_link_pxl=3)
plt.text(
    0.86,
    0.88,
    "N = " + str(df_plot.shape[0]),
    fontsize=11,
    transform=plt.gcf().transFigure,
    weight="bold",
)
plt.title(
    "Diffusion Coefficient Distribution \n(only within condensates)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("log10D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
fsave = "D Distribution.png"
plt.savefig(fsave, format="png")


######################################
# Distance to Edge Distribution with D splitting
plt.figure(figsize=(9, 4), dpi=200)
g = sns.histplot(
    data=df_plot,
    x="distance to edge (nm)",
    hue="tag",
    fill=True,
    stat="probability",
    alpha=0.3,
    common_norm=False,
    bins=30,
)
plt.title(
    "Distance to Edge Distribution", fontsize=13, fontweight="bold",
)
plt.xlabel("Distance to Edge (nm)", weight="bold")
g.legend_.set_title(None)
tags = df_plot["tag"].unique()
df0 = df_plot[df_plot["tag"] == tags[0]]
df1 = df_plot[df_plot["tag"] == tags[1]]
Gauss_fit_plot_text(df0["distance to edge (nm)"], 0, 0.78, 0.65)
Gauss_fit_plot_text(df1["distance to edge (nm)"], 1, 0.78, 0.61)
plt.tight_layout()
fsave = "Distance2Edge Distribution.png"
plt.savefig(fsave, format="png")


######################################
# Normalized distance to Edge Distribution with D splitting
plt.figure(figsize=(9, 4), dpi=200)
g = sns.histplot(
    data=df_plot,
    x="normalized distance",
    hue="tag",
    fill=True,
    stat="probability",
    alpha=0.3,
    common_norm=False,
    bins=30,
)
plt.title(
    "Normalized Distance Distribution", fontsize=13, fontweight="bold",
)
plt.xlabel("Distance to Edge, Normalized", weight="bold")
g.legend_.set_title(None)
tags = df_plot["tag"].unique()
df0 = df_plot[df_plot["tag"] == tags[0]]
df1 = df_plot[df_plot["tag"] == tags[1]]
Gauss_fit_plot_text(df0["normalized distance"], 0, 0.8, 0.65)
Gauss_fit_plot_text(df1["normalized distance"], 1, 0.8, 0.61)
plt.tight_layout()
fsave = "Normalized Distance2Edge Distribution.png"
plt.savefig(fsave, format="png")


######################################
# R Distribution with D splitting
plt.figure(figsize=(9, 4), dpi=200)
g = sns.histplot(
    data=df_plot,
    x="condensate R (nm)",
    hue="tag",
    fill=True,
    stat="probability",
    alpha=0.3,
    common_norm=False,
    bins=30,
)
plt.title(
    "Condensate R Distribution", fontsize=13, fontweight="bold",
)
plt.xlabel("Condensate R (nm)", weight="bold")
g.legend_.set_title(None)
tags = df_plot["tag"].unique()
df0 = df_plot[df_plot["tag"] == tags[0]]
df1 = df_plot[df_plot["tag"] == tags[1]]
Gauss_fit_plot_text(df0["condensate R (nm)"], 0, 0.76, 0.85)
Gauss_fit_plot_text(df1["condensate R (nm)"], 1, 0.76, 0.81)
plt.tight_layout()
fsave = "Condensate R Distribution.png"
plt.savefig(fsave, format="png")


######################################
# Negative Control: D Distribution with low frequency data
lst_fname2 = [
    f
    for f in os.listdir(folderpath)
    if f.endswith("RNA_linregress_D.csv") & f.startswith("low")
]
df_plot = pool_R2filter(lst_fname2, folderpath, R2threshold)
plt.figure(figsize=(9, 4), dpi=200)
g = sns.histplot(
    data=df_plot,
    x="log10D (um^2/s)",
    fill=True,
    stat="count",
    alpha=0.3,
    color="dimgray",
    bins=30,
)
axvline_bounds(s_per_frame=2, max_link_pxl=3)
DualGauss_fit_plot_text(df_plot["log10D (um^2/s)"])

data = df_plot["log10D (um^2/s)"]
counts, bins = np.histogram(data, bins=30)


def Gauss(x, A, x0, sigma):
    return A * np.exp((-1 / 2) * ((x - x0) ** 2 / sigma ** 2))


(A, x0, sigma), pcov = curve_fit(Gauss, (bins[1:] + bins[:-1]) / 2, counts,)
err_A, err_x0, err_sigma = np.sqrt(np.diag(pcov))
curve_x = np.arange(bins[0], bins[-1], 0.01)
curve_y = Gauss(curve_x, A, x0, sigma)
plt.plot(curve_x, curve_y, color=sns.color_palette()[7], linewidth=2)
plt.axvline(x=x0, color=sns.color_palette()[7], ls="--")
plt.text(
    0.76,
    0.7,
    "$\mu$ = " + str(round(x0, 2)) + "$\pm$" + str(round(err_x0, 2)),
    weight="bold",
    fontsize=11,
    color=sns.color_palette()[7],
    transform=plt.gcf().transFigure,
)
plt.text(
    0.86,
    0.88,
    "N = " + str(df_plot.shape[0]),
    fontsize=11,
    transform=plt.gcf().transFigure,
    weight="bold",
)
plt.title(
    "Diffusion Coefficient Distribution \n(only within condensates, 0.5 Hz)",
    fontsize=13,
    fontweight="bold",
)
plt.xlabel("log10D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
fsave = "D Distribution-lowfreq-fitting control.png"
plt.savefig(fsave, format="png")
