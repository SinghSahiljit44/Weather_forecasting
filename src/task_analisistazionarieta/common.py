import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data.preprocessing import load_daily

# statsmodels 0.15 announces a new return type for acf/adfuller. Passing result_object
# would break older versions (e.g. Colab), so only this notice is silenced
warnings.filterwarnings("ignore", message=r".*result_object.*", category=FutureWarning)

TASK_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TASK_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
STL_PATH = RESULTS_DIR / "stl_components.parquet"

# The two forecasting targets of the project
SERIES = {
    "t2m_mean": "Temperatura media giornaliera",
    "tp_mm": "Precipitazione giornaliera",
}
UNITS = {"t2m_mean": "°C", "tp_mm": "mm"}

# Daily data: one seasonal cycle is one year
PERIOD = 365
ALPHA = 0.05

# Validated categorical slots (light surface) and recessive chrome
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.prop_cycle": matplotlib.cycler(color=[BLUE, ORANGE, AQUA]),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "font.size": 9,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.2,
        }
    )


def load_series():
    """Daily target series with an explicit daily frequency (needed by STL)."""
    return load_daily()[list(SERIES)].asfreq("D")


def load_stl():
    return pd.read_parquet(STL_PATH)


def stl_component(stl, variable, component):
    return stl[f"{variable}_{component}"]


def save_table(df, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RESULTS_DIR / f"{name}.parquet")


def save_figure(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def quiet(fn, *args, **kwargs):
    """Run fn and return (result, warnings raised), so callers can record them."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught]
