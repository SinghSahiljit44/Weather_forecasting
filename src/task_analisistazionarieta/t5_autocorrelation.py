"""Task 5 - Analisi dell'autocorrelazione: ACF e PACF."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

from src.task_analisistazionarieta import t2_stl
from src.task_analisistazionarieta.common import (
    ALPHA,
    BLUE,
    INK_SECONDARY,
    SERIES,
    load_series,
    save_figure,
    save_table,
    setup_style,
    stl_component,
)

TITLE = "Task 5 - ACF e PACF"
SHORT_LAGS = 40
# Long enough to show the yearly peaks at 365 and 730 days
LONG_LAGS = 800
PACF_METHOD = "ywm"


def correlogram(x, function, nlags):
    if function == "acf":
        values, confint = acf(x, nlags=nlags, alpha=ALPHA, fft=True)
    else:
        values, confint = pacf(x, nlags=nlags, alpha=ALPHA, method=PACF_METHOD)
    # confint is centred on the estimate; its half width is the band around zero
    band = (confint[:, 1] - confint[:, 0]) / 2
    return pd.DataFrame(
        {"lag": np.arange(len(values)), "valore": values, "banda": band, "significativo": np.abs(values) > band}
    )


def inputs(series, stl, variable):
    return {"originale": series[variable], "residuo STL": stl_component(stl, variable, "resid")}


def build_table(series, stl):
    frames = []
    for variable in SERIES:
        for label, x in inputs(series, stl, variable).items():
            for function, nlags in [("acf", LONG_LAGS), ("pacf", SHORT_LAGS)]:
                frame = correlogram(x.to_numpy(), function, nlags)
                frame.insert(0, "funzione", function)
                frame.insert(0, "serie", label)
                frame.insert(0, "variable", variable)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(table):
    rows = []
    for (variable, label), group in table.groupby(["variable", "serie"], sort=False):
        a = group[group.funzione == "acf"].set_index("lag")
        p = group[group.funzione == "pacf"].set_index("lag")
        # First lag at which the ACF falls inside the band: how long memory lasts
        inside = a.index[(a.index > 0) & ~a.significativo]
        rows.append(
            {
                "variable": variable,
                "serie": label,
                "acf_lag1": a.valore[1],
                "acf_lag7": a.valore[7],
                "acf_lag30": a.valore[30],
                "acf_lag365": a.valore[365],
                "acf_lag730": a.valore[730],
                "primo_lag_acf_non_significativo": int(inside[0]) if len(inside) else None,
                "pacf_lag_significativi_1_10": [int(k) for k in p.index[(p.index >= 1) & (p.index <= 10) & p.significativo]],
                "pacf_lag1": p.valore[1],
                "pacf_lag2": p.valore[2],
            }
        )
    return pd.DataFrame(rows)


def plot(series, stl, variable):
    data = inputs(series, stl, variable)
    raw, resid = data["originale"].to_numpy(), data["residuo STL"].to_numpy()
    style = {"color": BLUE, "markersize": 3}
    thin = {"colors": BLUE, "linewidths": 0.8}

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8))
    plot_acf(raw, ax=axes[0, 0], lags=LONG_LAGS, fft=True, title="ACF serie originale (lag 0-800)",
             vlines_kwargs={"colors": BLUE, "linewidths": 0.4}, color=BLUE, markersize=0.8)
    plot_acf(resid, ax=axes[0, 1], lags=SHORT_LAGS, fft=True, title="ACF residuo STL (lag 0-40)",
             vlines_kwargs=thin, **style)
    plot_pacf(raw, ax=axes[1, 0], lags=SHORT_LAGS, method=PACF_METHOD, title="PACF serie originale (lag 0-40)",
              vlines_kwargs=thin, **style)
    plot_pacf(resid, ax=axes[1, 1], lags=SHORT_LAGS, method=PACF_METHOD, title="PACF residuo STL (lag 0-40)",
              vlines_kwargs=thin, **style)

    long_acf = acf(raw, nlags=LONG_LAGS, fft=True)
    for lag, label in [(365, "1 anno"), (730, "2 anni")]:
        axes[0, 0].annotate(label, (lag, long_acf[lag]), xytext=(0, 8), textcoords="offset points",
                            ha="center", color=INK_SECONDARY, fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel("lag (giorni)")
        # statsmodels centres its title: move it to the left edge
        title = ax.get_title()
        ax.set_title("")
        ax.set_title(title, loc="left", fontsize=9)
    fig.suptitle(f"{SERIES[variable]} - autocorrelazione", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, f"t5_acf_pacf_{variable}")


def run(verbose=True):
    series = load_series()
    stl = t2_stl.get_components()
    table = build_table(series, stl)
    summary = summarize(table)
    save_table(table, "acf_pacf")
    save_table(summary, "acf_pacf_summary")
    for variable in SERIES:
        plot(series, stl, variable)

    if verbose:
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return summary


if __name__ == "__main__":
    setup_style()
    run()
