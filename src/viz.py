import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import pandas as pd

PAGE_WIDTH = 6.99  # in
ROW_HEIGHT = 2.5  # in
COLORS = sns.color_palette("colorblind")


def reset_fonts(style="small", SMALL_SIZE=None, MEDIUM_SIZE=None, BIGGER_SIZE=None):
    if style == "big":
        SMALL_SIZE = 22
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 26

    if SMALL_SIZE is None:
        SMALL_SIZE = 16

    if MEDIUM_SIZE is None:
        MEDIUM_SIZE = 18

    if BIGGER_SIZE is None:
        BIGGER_SIZE = 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title


def set_plots(style=""):
    """
    Set up custom plotting style.

    Parameters
    ----------
    style : str in ["analysis", "notebook", ""]
        style to use

    Returns
    -------
    COLORS
    PAGE_WIDTH
    ROW_HEIGHT
    """
    pd.plotting.register_matplotlib_converters()
    plt.rcParams.update(
        {
            "figure.figsize": [PAGE_WIDTH / 2, ROW_HEIGHT],
            "grid.color": "k",
            "axes.grid": True,
            "font.size": 10,
            "axes.prop_cycle": cycler("color", COLORS),
        }
    )
    if style == "":
        plt.rcParams.update(
            {
                "grid.linestyle": ":",
                "grid.linewidth": 0.5,
                "figure.dpi": 200,
            }
        )
    elif style == "analysis":
        plt.rcParams.update(
            {
                "grid.linewidth": 0.2,
                "grid.color": "grey",
                "figure.dpi": 300,
                "xtick.labelsize": 5,
                "ytick.labelsize": 5,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "savefig.transparent": True,
                "legend.fontsize": 7,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "savefig.bbox": "tight",
                "legend.markerscale": 3.0,
            }
        )
    elif style == "notebook":
        plt.rcParams.update(
            {
                "figure.figsize": [PAGE_WIDTH, ROW_HEIGHT],
                "axes.grid": True,
                "grid.linewidth": 0.2,
                "grid.color": "grey",
                "figure.dpi": 300,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "savefig.transparent": True,
                "legend.fontsize": 7,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "legend.markerscale": 3.0,
            }
        )

    return (COLORS, PAGE_WIDTH, ROW_HEIGHT)


COLORS, PAGE_WIDTH, ROW_HEIGHT = set_plots()


def plotOperations(df, START_DATE=None, END_DATE=None, save_path=None):
    """
    Plot the number of operations per operation type
    :param df: DataFrame with the operations
    """
    if START_DATE is not None and END_DATE is not None:
        df = df.loc[START_DATE:END_DATE]
    f, ax = plt.subplots(3, 1, figsize=(PAGE_WIDTH, 3 * ROW_HEIGHT))
    ax[0].plot(df["load Batt"], label="Battery Load (MW)")
    ax[0].plot(df["load EVs"], label="EVs Load (MW)")
    ax[0].plot(df["load"], label="Total Load (MW)", linestyle="--", alpha=0.5)
    ax[0].set_title("Station Load")
    ax[0].legend(loc="upper right")
    ax[0].set_ylabel("Load (MW)")
    ax[1].plot(100 * df["SOC"], label="Battery SOC (%)", color=COLORS[3])
    ax[1].set_title("Battery SOC")
    ax[1].legend(loc="upper right")
    ax[1].set_ylabel("SOC (%)")
    ax2 = ax[1].twinx()
    ax2.plot(
        df["price"], label="Price ($/MWh)", color="grey", alpha=0.5, linestyle="--"
    )
    ax2.set_ylabel("Price ($/MWh)", color="grey")
    ax2.legend(loc="upper left")

    ax[2].plot(df["newEVs"], label="Vehicles Entering")
    ax[2].set_title("Vehicles Entering the station")
    ax[2].set_ylabel("Number of Vehicles")
    ax[2].legend(loc="upper right")
    f.tight_layout()
    f.autofmt_xdate()
    if save_path is not None:
        plt.savefig(save_path)
