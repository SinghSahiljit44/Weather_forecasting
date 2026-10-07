"""Task 2 - Decomposizione STL: trend, componente stagionale, residuo."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from src.task_analisistazionarieta.common import (
    BLUE,
    PERIOD,
    SERIES,
    STL_PATH,
    UNITS,
    load_series,
    save_figure,
    save_table,
    setup_style,
)

TITLE = "Task 2 - Decomposizione STL"
COMPONENTS = ["observed", "trend", "seasonal", "resid"]
PANEL_TITLES = ["Serie osservata", "Trend", "Componente stagionale", "Residuo"]

# The default seasonal=7 averages each calendar day over ~7 years only, so the seasonal
# component keeps ~1 °C of day-to-day noise and, for rain, absorbs single storms.
# 37 spans the whole 36-year record: a stable seasonal shape that can still drift slowly.
SEASONAL = 37
# robust=True would treat every rainy day as an outlier (the median day is dry)
ROBUST = False
SENSITIVITY_SEASONAL = [7, 13, 25, 37, 73]
SMOOTH_DAYS = 31


def decompose(series, period=PERIOD, seasonal=SEASONAL, robust=ROBUST):
    return STL(series, period=period, seasonal=seasonal, robust=robust).fit()


def strength(result):
    """Wang, Smith & Hyndman (2006): 0 = component absent, 1 = component dominates."""
    t, s, r = result.trend, result.seasonal, result.resid
    return {
        "forza_trend": max(0.0, 1 - r.var() / (t + r).var()),
        "forza_stagionale": max(0.0, 1 - r.var() / (s + r).var()),
    }


def seasonal_amplitude(seasonal, smooth=True):
    """Mean yearly max-min of the seasonal component, smoothed so noise does not inflate it."""
    if smooth:
        seasonal = seasonal.rolling(SMOOTH_DAYS, center=True, min_periods=1).mean()
    return seasonal.groupby(seasonal.index.year).agg(lambda x: x.max() - x.min())


def summarize(label, unit, result, per_decade, smooth=True):
    y, t, s, r = result.observed, result.trend, result.seasonal, result.resid
    years = y.index.year
    # STL trend is least reliable at the edges, so compare whole years
    first, last = years.min(), years.max()
    amplitude = seasonal_amplitude(s, smooth)
    slope = np.polyfit(np.arange(len(t)), t.to_numpy(), 1)[0]
    return {
        "serie": label,
        "unita": unit,
        **strength(result),
        "quota_var_trend": t.var() / y.var(),
        "quota_var_stagionale": s.var() / y.var(),
        "quota_var_residuo": r.var() / y.var(),
        "trend_primo_anno": t[years == first].mean(),
        "trend_ultimo_anno": t[years == last].mean(),
        "trend_variazione_totale": t[years == last].mean() - t[years == first].mean(),
        "trend_pendenza_per_decennio": slope * per_decade,
        "ampiezza_stagionale_media": amplitude.mean(),
        "ampiezza_stagionale_1990_1994": amplitude.loc[first : first + 4].mean(),
        "ampiezza_stagionale_2021_2025": amplitude.loc[last - 4 : last].mean(),
    }


def sensitivity(series):
    rows = []
    for variable in SERIES:
        for seasonal in SENSITIVITY_SEASONAL:
            for robust in [False, True]:
                result = decompose(series[variable], seasonal=seasonal, robust=robust)
                s = result.seasonal
                rows.append(
                    {
                        "variable": variable,
                        "seasonal": seasonal,
                        "robust": robust,
                        **strength(result),
                        "quota_var_stagionale": s.var() / result.observed.var(),
                        "rumore_stagionale_sd": (s - s.rolling(SMOOTH_DAYS, center=True, min_periods=1).mean()).std(),
                        "ampiezza_stagionale_media": seasonal_amplitude(s).mean(),
                        "residuo_sd": result.resid.std(),
                    }
                )
    return pd.DataFrame(rows)


def plot(variable, result):
    unit = UNITS[variable]
    fig, axes = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True)
    for ax, component, title in zip(axes, COMPONENTS, PANEL_TITLES):
        values = getattr(result, component)
        width = 1.6 if component == "trend" else 0.35
        ax.plot(values.index, values, color=BLUE, linewidth=width)
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
    fig.suptitle(
        f"{SERIES[variable]} - decomposizione STL (period={PERIOD}, seasonal={SEASONAL})",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, f"t2_stl_{variable}")


def components_frame(results):
    columns = {
        f"{variable}_{component}": getattr(result, component)
        for variable, result in results.items()
        for component in COMPONENTS
    }
    frame = pd.DataFrame(columns)
    frame.index.name = "date"
    return frame


def get_components():
    """STL components from disk, computing them first if this task has not run yet."""
    if not STL_PATH.exists():
        run(verbose=False)
    return pd.read_parquet(STL_PATH)


def run(verbose=True):
    series = load_series()
    results = {v: decompose(series[v]) for v in SERIES}
    save_table(components_frame(results), "stl_components")

    rows = [summarize(f"{v} (giornaliera)", UNITS[v], r, per_decade=3652.5) for v, r in results.items()]
    # At monthly scale the rain seasonality is no longer buried under daily noise
    monthly_rain = series["tp_mm"].resample("MS").sum()
    rows.append(
        summarize(
            "tp_mm (totali mensili)",
            "mm/mese",
            decompose(monthly_rain, period=12),
            per_decade=120,
            smooth=False,
        )
    )
    summary = pd.DataFrame(rows).set_index("serie")
    save_table(summary, "stl_summary")
    save_table(sensitivity(series), "stl_sensitivity")

    for variable, result in results.items():
        plot(variable, result)

    if verbose:
        print(summary.T.round(3).to_string())
    return summary


if __name__ == "__main__":
    setup_style()
    run()
