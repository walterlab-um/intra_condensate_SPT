from os.path import join, dirname, basename
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation

sns.set(color_codes=True, style="white")

pd.options.mode.chained_assignment = None  # default='warn'
color_palette = ["#f4b942", "#a1dab4", "#41b6c4", "#225ea8"]

os.chdir(folderpath)
df_noTotR = pd.read_csv("EffectiveD-alpha-alltracks-noTotR.csv")
# df_hela = pd.read_csv("EffectiveD-alpha-alltracks-FL-noDex-Hela.csv")
# df_cerebral = pd.read_csv("EffectiveD-alpha-alltracks-FL-noDex-Cerebral.csv")
df_spinal = pd.read_csv("EffectiveD-alpha-alltracks-Spinal.csv")

um_per_pixel = 0.117
s_per_frame = 0.02

# calculate error bounds
static_err = 0.016
um_per_pxl = 0.117
link_max = 3
log10D_low = np.log10(static_err**2 / (4 * (s_per_frame)))
log10D_high = np.log10((um_per_pxl * link_max) ** 2 / (4 * (s_per_frame)))


def histogrameplot(df, key_threshold, key_value, binrange, colornum, label):
    global color_palette
    sns.histplot(
        data=df[df[key_threshold] > 0.7][key_value],
        stat="density",
        bins=50,
        color=color_palette[colornum],
        alpha=0.5,
        binrange=binrange,
        label=label,
        kde=True,
        line_kws={"linewidth": 3},
    )


def histo_accumulate_plot(df, key_threshold, key_value, binrange, colornum, label):
    global color_palette
    sns.histplot(
        data=df[df[key_threshold] > 0.7][key_value],
        stat="density",
        element="step",
        fill=False,
        cumulative=True,
        bins=50,
        color=color_palette[colornum],
        binrange=binrange,
        label=label,
        line_kws={"linewidth": 3},
    )


#######################################
# log10D loglog or linear

# linear
plt.figure(figsize=(9, 4), dpi=200)
key_threshold = "R2_loglog"
key_value = "log10D_linear"
binrange = (log10D_low - 1.5, log10D_high + 1.5)
histogrameplot(df_noTotR, key_threshold, key_value, binrange, 0, "no Total RNA")
# histogrameplot(df_hela, key_threshold, key_value, binrange, 1, "Hela")
# histogrameplot(df_cerebral, key_threshold, key_value, binrange, 2, "Cerebral Cortex")
histogrameplot(df_spinal, key_threshold, key_value, binrange, 3, "Spinal Cord")
plt.axvspan(
    log10D_low - 1.5, log10D_low, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.axvspan(
    log10D_high, log10D_high + 1.5, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.xlim(log10D_low - 1.5, log10D_high + 1.5)
plt.legend(loc=1)
plt.title("FL mRNA, no Dextran, linear fitting", weight="bold")
plt.xlabel("log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
plt.savefig("Effect of TotR-log10D_linear.png", format="png")
plt.close()

# linear accumulate
plt.figure(figsize=(9, 4), dpi=200)
histo_accumulate_plot(df_noTotR, key_threshold, key_value, binrange, 0, "no Total RNA")
# histo_accumulate_plot(df_hela, key_threshold, key_value, binrange, 1, "Hela")
# histo_accumulate_plot(
#     df_cerebral, key_threshold, key_value, binrange, 2, "Cerebral Cortex"
# )
histo_accumulate_plot(df_spinal, key_threshold, key_value, binrange, 3, "Spinal Cord")
plt.xlim(log10D_low - 1.5, log10D_high + 1.5)
plt.legend(loc=2)
plt.title("FL mRNA, no Dextran, linear fitting", weight="bold")
plt.xlabel("log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
plt.savefig("Effect of TotR-log10D_linear_accumulate.png", format="png")
plt.close()

# log log
plt.figure(figsize=(9, 4), dpi=200)
key_threshold = "R2_loglog"
key_value = "log10D_loglog"
binrange = (log10D_low - 1.5, log10D_high + 1.5)
histogrameplot(df_noTotR, key_threshold, key_value, binrange, 0, "no Total RNA")
# histogrameplot(df_hela, key_threshold, key_value, binrange, 1, "Hela")
# histogrameplot(df_cerebral, key_threshold, key_value, binrange, 2, "Cerebral Cortex")
histogrameplot(df_spinal, key_threshold, key_value, binrange, 3, "Spinal Cord")
plt.axvspan(
    log10D_low - 1.5, log10D_low, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.axvspan(
    log10D_high, log10D_high + 1.5, facecolor="dimgray", alpha=0.2, edgecolor="none"
)
plt.xlim(log10D_low - 1.5, log10D_high + 1.5)
plt.legend(loc=1)
plt.title("FL mRNA, no Dextran, log-log fitting", weight="bold")
plt.xlabel("log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
plt.savefig("Effect of TotR-log10D_loglog.png", format="png")
plt.close()

# log log accumulate
plt.figure(figsize=(9, 4), dpi=200)
histo_accumulate_plot(df_noTotR, key_threshold, key_value, binrange, 0, "no Total RNA")
# histo_accumulate_plot(df_hela, key_threshold, key_value, binrange, 1, "Hela")
# histo_accumulate_plot(
#     df_cerebral, key_threshold, key_value, binrange, 2, "Cerebral Cortex"
# )
histo_accumulate_plot(df_spinal, key_threshold, key_value, binrange, 3, "Spinal Cord")
plt.xlim(log10D_low - 1.5, log10D_high + 1.5)
plt.legend(loc=2)
plt.title("FL mRNA, no Dextran, log-log fitting", weight="bold")
plt.xlabel("log$_{10}$D ($\mu$m^2/s)", weight="bold")
plt.tight_layout()
plt.savefig("Effect of TotR-log10D_loglog_accumulate.png", format="png")
plt.close()

#######################################
# alpha
plt.figure(figsize=(9, 4), dpi=200)
key_threshold = "R2_loglog"
key_value = "alpha"
binrange = (0, 1.5)
histogrameplot(df_noTotR, key_threshold, key_value, binrange, 0, "no Total RNA")
# histogrameplot(df_hela, key_threshold, key_value, binrange, 1, "Hela")
# histogrameplot(df_cerebral, key_threshold, key_value, binrange, 2, "Cerebral Cortex")
histogrameplot(df_spinal, key_threshold, key_value, binrange, 3, "Spinal Cord")
plt.xlim(0, 1.5)
plt.legend(loc=1)
plt.title("Anomalous Exponent", weight="bold")
plt.xlabel("$\u03B1$", weight="bold")
plt.tight_layout()
plt.savefig("Effect of TotR-alpha.png", format="png")
plt.close()

# accumulate
plt.figure(figsize=(9, 4), dpi=200)
histo_accumulate_plot(df_noTotR, key_threshold, key_value, binrange, 0, "no Total RNA")
# histo_accumulate_plot(df_hela, key_threshold, key_value, binrange, 1, "Hela")
# histo_accumulate_plot(
#     df_cerebral, key_threshold, key_value, binrange, 2, "Cerebral Cortex"
# )
histo_accumulate_plot(df_spinal, key_threshold, key_value, binrange, 3, "Spinal Cord")
plt.xlim(0, 1.5)
plt.legend(loc=1)
plt.title("Anomalous Exponent", weight="bold")
plt.xlabel("$\u03B1$", weight="bold")
plt.tight_layout()
plt.savefig("Effect of TotR-alpha_accumulate.png", format="png")
plt.close()

#######################################
# percentage of constrained molecule as of alpha<0.7, and immobile
lst_tag = []
lst_FOVname = []
lst_fraction_constrained = []
lst_fraction_immobile = []


def calculate_contrained_fraction(df_in, tag):
    global lst_tag, lst_FOVname, lst_fraction_constrained, lst_fraction_immobile
    allFOVs = df_in.filename.unique()
    for FOVname in allFOVs:
        lst_tag.append(tag)
        lst_FOVname.append(FOVname)
        df_currentFOV = df_in[df_in.filename == FOVname]
        # calculate constrained fraction
        all_alpha_value = np.array(
            df_currentFOV[df_currentFOV["R2_loglog"] > 0.7].alpha
        )
        fraction_constrained = len(all_alpha_value[all_alpha_value < 0.7]) / len(
            all_alpha_value
        )
        lst_fraction_constrained.append(fraction_constrained)
        # calculate immobile fraction
        df_immobile = df_currentFOV[
            df_currentFOV["slope_linear"] < 10**log10D_low * (8 / 3)
        ]
        lst_fraction_immobile.append(df_immobile.shape[0] / df_currentFOV.shape[0])


calculate_contrained_fraction(df_noTotR, "no Total RNA")
# calculate_contrained_fraction(df_hela, "Hela")
# calculate_contrained_fraction(df_cerebral, "Cerebral Cortex")
calculate_contrained_fraction(df_spinal, "Spinal Cord")

df_plot = pd.DataFrame(
    {
        "tag": lst_tag,
        "FOVname": lst_FOVname,
        "constrained_percent": np.array(lst_fraction_constrained) * 100,
        "immobile_percent": np.array(lst_fraction_immobile) * 100,
    },
    dtype=object,
)
df_plot = df_plot.astype({"constrained_percent": float, "immobile_percent": float})
df_plot.to_csv("constrained_and_immobile_percent.csv", index=False)

plt.figure(figsize=(4, 6), dpi=200)
order = ["no Total RNA", "Spinal Cord"]
box_pairs = [
    ("no Total RNA", "Spinal Cord"),
]
ax = sns.boxplot(
    data=df_plot,
    x="tag",
    y="constrained_percent",
    width=0.5,
    order=order,
    palette=[color_palette[0], color_palette[3]],
)
ax = sns.stripplot(
    data=df_plot,
    x="tag",
    y="constrained_percent",
    color=".25",
    order=order,
)
test_results = add_stat_annotation(
    ax,
    data=df_plot,
    x="tag",
    y="constrained_percent",
    order=order,
    box_pairs=box_pairs,
    test="Mann-Whitney",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.title("Constrained Percent", weight="bold")
plt.ylabel("% $\u03B1$ < 0.7, per FOV", weight="bold")
ax.xaxis.set_tick_params(labelsize=17, labelrotation=30)
plt.xlabel("")
plt.tight_layout()
plt.savefig("Effect of TotR-constrained percent.png", format="png")
plt.close()

plt.figure(figsize=(4, 6), dpi=200)
ax = sns.boxplot(
    data=df_plot,
    x="tag",
    y="immobile_percent",
    width=0.5,
    order=order,
    palette=[color_palette[0], color_palette[3]],
)
ax = sns.stripplot(
    data=df_plot,
    x="tag",
    y="immobile_percent",
    color=".25",
    order=order,
)
test_results = add_stat_annotation(
    ax,
    data=df_plot,
    x="tag",
    y="immobile_percent",
    order=order,
    box_pairs=box_pairs,
    test="Mann-Whitney",
    comparisons_correction=None,
    text_format="star",
    loc="inside",
    verbose=2,
)
plt.title("Immobile Percent", weight="bold")
plt.ylabel(r"% linear slope < D lower bound * 8/3, per FOV", weight="bold")
ax.xaxis.set_tick_params(labelsize=17, labelrotation=30)
plt.xlabel("")
plt.tight_layout()
plt.savefig("Effect of TotR-immobile percent.png", format="png")
plt.close()
