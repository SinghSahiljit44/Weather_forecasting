"""Task 3 - Analisi del trend con il test di Mann-Kendall."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk

from src.task_analisistazionarieta import t2_stl
from src.task_analisistazionarieta.common import (
    BLUE,
    ORANGE,
    SERIES,
    load_series,
    save_figure,
    save_table,
    setup_style,
    stl_component,
)

TITLE = "Task 3 - Test di Mann-Kendall"

UNIT_PER_DECADE = {
    "t2m_mean": {"giornaliera": "°C", "mensile": "°C", "annuale": "°C"},
    "tp_mm": {"giornaliera": "mm/giorno", "mensile": "mm/mese", "annuale": "mm/anno"},
}


def aggregate(series, variable, rule):
    # Rain is a total, temperature an average
    resampled = series.resample(rule)
    return resampled.sum() if variable == "tp_mm" else resampled.mean()


def row(variable, series_label, scale, test_name, result, n, per_decade):
    return {
        "variable": variable,
        "serie": series_label,
        "scala": scale,
        "test": test_name,
        "n": n,
        "trend": result.trend,
        "significativo": bool(result.h),
        "p": float(result.p),
        "z": float(result.z),
        "tau": float(result.Tau),
        "sen_slope": float(result.slope),
        "pendenza_per_decennio": float(result.slope) * per_decade,
        "unita": f"{UNIT_PER_DECADE[variable][scale]} per decennio",
    }


def run_tests(series, stl):
    rows = []
    for variable in SERIES:
        daily = series[variable]
        deseasonalized = stl_component(stl, variable, "observed") - stl_component(stl, variable, "seasonal")
        monthly = aggregate(daily, variable, "MS")
        annual = aggregate(daily, variable, "YS")

        # As in the assignment: valid only for independent data, shown for comparison
        rows.append(row(variable, "originale", "giornaliera", "original_test",
                        mk.original_test(daily.to_numpy()), len(daily), 3652.5))
        # Same deseasonalized input, with and without the autocorrelation correction:
        # Hamed & Rao inflate the variance of S, so the gap isolates the effect of persistence
        rows.append(row(variable, "destagionalizzata (STL)", "giornaliera", "original_test",
                        mk.original_test(deseasonalized.to_numpy()), len(daily), 3652.5))
        rows.append(row(variable, "destagionalizzata (STL)", "giornaliera", "hamed_rao_modification_test",
                        mk.hamed_rao_modification_test(deseasonalized.to_numpy()), len(daily), 3652.5))
        # Compares each month only with the same month of other years: seasonality drops out
        rows.append(row(variable, "originale", "mensile", "seasonal_test (period=12)",
                        mk.seasonal_test(monthly.to_numpy(), period=12), len(monthly), 10))
        rows.append(row(variable, "originale", "annuale", "original_test",
                        mk.original_test(annual.to_numpy()), len(annual), 10))
    return pd.DataFrame(rows)


def plot(series):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = {"t2m_mean": ("Temperatura media annua", "°C"), "tp_mm": ("Precipitazione annua", "mm")}
    for ax, variable in zip(axes, SERIES):
        annual = aggregate(series[variable], variable, "YS")
        result = mk.original_test(annual.to_numpy())
        years = annual.index.year
        sen_line = result.intercept + result.slope * np.arange(len(annual))
        title, unit = labels[variable]

        verdict = "significativo" if result.h else "non significativo"
        ax.plot(years, annual, "o-", color=BLUE, markersize=4, linewidth=1, label="valore annuale")
        ax.plot(years, sen_line, color=ORANGE, linewidth=2,
                label=f"pendenza di Sen (Mann-Kendall p = {result.p:.1g}, {verdict})")
        ax.set_title(f"{title}: {result.slope * 10:+.2f} {unit}/decennio", loc="left")
        ax.set_ylabel(unit)
        # Below the axes: inside, it would sit on the wettest years
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    save_figure(fig, "t3_mann_kendall_annuale")


def run(verbose=True):
    series = load_series()
    stl = t2_stl.get_components()
    table = run_tests(series, stl)
    save_table(table, "mann_kendall")
    plot(series)

    if verbose:
        cols = ["variable", "serie", "scala", "test", "n", "trend", "p", "z", "pendenza_per_decennio", "unita"]
        print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return table


if __name__ == "__main__":
    setup_style()
    run()
