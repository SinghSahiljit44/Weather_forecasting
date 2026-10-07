"""Task 7 (opzionale) - Cambiamenti strutturali con PELT (ruptures)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt

from src.task_analisistazionarieta.common import (
    BLUE,
    INK_MUTED,
    INK_SECONDARY,
    ORANGE,
    SERIES,
    SURFACE,
    UNITS,
    load_series,
    save_figure,
    save_table,
    setup_style,
)

TITLE = "Task 7 - Cambiamenti strutturali (PELT)"
PENALTIES = [1, 2, 3, 5, 10, 20, 50]
DEFAULT_PENALTY = 10
MODEL = "rbf"


def monthly_anomalies(series, variable):
    """Monthly values minus their calendar-month mean: the seasonal cycle would be
    found as a 'change' every spring and autumn otherwise."""
    resampled = series.resample("MS")
    monthly = resampled.sum() if variable == "tp_mm" else resampled.mean()
    return monthly - monthly.groupby(monthly.index.month).transform("mean")


def remove_linear_trend(x):
    t = np.arange(len(x))
    slope, intercept = np.polyfit(t, x.to_numpy(), 1)
    return x - (intercept + slope * t)


def inputs(series, variable):
    anomalies = monthly_anomalies(series[variable], variable)
    # A steady trend looks like a staircase to a change-point method: removing it
    # tells real jumps apart from the warming itself
    return {"anomalie mensili": anomalies, "anomalie mensili senza trend lineare": remove_linear_trend(anomalies)}


def detect(x, penalty):
    algo = rpt.Pelt(model=MODEL).fit(x.to_numpy().reshape(-1, 1))
    # predict() returns the end index of each segment; the last one is len(x)
    return algo.predict(pen=penalty)[:-1]


def segment_means(x, breakpoints):
    edges = [0, *breakpoints, len(x)]
    means = pd.Series(np.nan, index=x.index)
    for start, end in zip(edges[:-1], edges[1:]):
        means.iloc[start:end] = x.iloc[start:end].mean()
    return means


def run_detection(series):
    rows = []
    for variable in SERIES:
        for label, x in inputs(series, variable).items():
            for penalty in PENALTIES:
                breakpoints = detect(x, penalty)
                edges = [0, *breakpoints, len(x)]
                rows.append(
                    {
                        "variable": variable,
                        "serie": label,
                        "penalita": penalty,
                        "n_cambiamenti": len(breakpoints),
                        "date": ", ".join(x.index[b].strftime("%Y-%m") for b in breakpoints),
                        "medie_segmenti": ", ".join(
                            f"{x.iloc[s:e].mean():+.2f}" for s, e in zip(edges[:-1], edges[1:])
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot(series):
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(11, 6.2))
    for ax, variable in zip(axes, SERIES):
        x = inputs(series, variable)["anomalie mensili"]
        breakpoints = detect(x, DEFAULT_PENALTY)
        ax.plot(x.index, x, color=BLUE, linewidth=0.7, label="anomalia mensile")
        ax.plot(x.index, segment_means(x, breakpoints), color=ORANGE, linewidth=2, label="media del segmento")
        for b in breakpoints:
            ax.axvline(x.index[b], color=INK_MUTED, linewidth=1)
            ax.annotate(x.index[b].strftime("%Y-%m"), (x.index[b], 1), xycoords=("data", "axes fraction"),
                        xytext=(3, -12), textcoords="offset points", color=INK_SECONDARY, fontsize=8)
        n = len(breakpoints)
        found = "nessun cambiamento" if n == 0 else f"{n} cambiament{'o' if n == 1 else 'i'}"
        ax.set_title(f"{SERIES[variable]}: anomalie mensili, PELT rbf, pen={DEFAULT_PENALTY} - {found}", loc="left")
        ax.set_ylabel(UNITS[variable] if variable != "tp_mm" else "mm/mese")
        ax.legend(loc="lower right", ncols=2, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, "t7_change_points")


def run(verbose=True):
    series = load_series()
    table = run_detection(series)
    save_table(table, "change_points")
    plot(series)

    if verbose:
        print(table.drop(columns="medie_segmenti").to_string(index=False))
    return table


if __name__ == "__main__":
    setup_style()
    run()
