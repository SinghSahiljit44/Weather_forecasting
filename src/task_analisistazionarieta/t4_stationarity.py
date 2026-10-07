"""Task 4 - Test di stazionarietà: ADF e KPSS sulla serie originale."""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.task_analisistazionarieta.common import ALPHA, PERIOD, SERIES, load_series, quiet, save_table, setup_style

TITLE = "Task 4 - Test di stazionarietà (ADF, KPSS)"
KPSS_LAGS = [5, 10, 20, 40, "auto", 150, 365]


def adf_test(x):
    # H0: unit root (non stationary)
    stat, p, lags, nobs, crit, _ = adfuller(x, autolag="AIC")
    return {"adf_stat": stat, "adf_p": p, "adf_lags": lags, "adf_crit_5%": crit["5%"]}


def kpss_test(x, regression="c", nlags="auto"):
    # H0: stationary around a constant ("c") or around a linear trend ("ct")
    (stat, p, lags, crit), caught = quiet(kpss, x, regression=regression, nlags=nlags)
    # statsmodels interpolates p from a table covering [0.01, 0.10] only
    bound = "="
    if any("smaller" in w for w in caught):
        bound = "<"
    elif any("greater" in w for w in caught):
        bound = ">"
    key = f"kpss_{regression}"
    return {f"{key}_stat": stat, f"{key}_p": p, f"{key}_p_bound": bound, f"{key}_lags": lags}


def verdict(adf_p, kpss_p):
    no_unit_root = adf_p < ALPHA
    kpss_stationary = kpss_p > ALPHA
    if no_unit_root and kpss_stationary:
        return "stazionaria"
    if not no_unit_root and not kpss_stationary:
        return "non stazionaria (radice unitaria)"
    if no_unit_root and not kpss_stationary:
        # Not necessarily deterministic: ADF has little power against a small random walk
        return "contrastante (ADF: no radice unitaria, KPSS: non stazionaria)"
    return "inconcludente"


def test_series(x, variable, label):
    x = pd.Series(x).dropna().to_numpy()
    result = {"variable": variable, "serie": label, "n": len(x)}
    result |= adf_test(x)
    result |= kpss_test(x, "c")
    result |= kpss_test(x, "ct")
    result["verdetto"] = verdict(result["adf_p"], result["kpss_c_p"])
    return result


def kpss_lag_sensitivity(series):
    """The KPSS verdict depends on the bandwidth of the long-run variance estimate."""
    rows = []
    for variable in SERIES:
        for nlags in KPSS_LAGS:
            r = kpss_test(series[variable].to_numpy(), "c", nlags)
            rows.append({"variable": variable, "nlags": str(nlags), "lags_usati": r["kpss_c_lags"],
                         "stat": r["kpss_c_stat"], "p": r["kpss_c_p"], "p_bound": r["kpss_c_p_bound"]})
    return pd.DataFrame(rows)


def synthetic_check(n, seed=0):
    """What can ADF and KPSS actually see? Known components on a series as long as ours."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    noise = rng.normal(0, 2.7, n)  # sd of the temperature anomaly
    season = 10 * np.sin(2 * np.pi * t / PERIOD)
    cases = {
        "rumore bianco": noise,
        "stagionalità + rumore": season + noise,
        "stagionalità + rumore + trend (+1,4 in 36 anni)": season + noise + 1.4 * t / n,
        "stagionalità + rumore + passeggiata aleatoria": season + noise + np.cumsum(rng.normal(0, 0.3, n)),
    }
    return pd.DataFrame([test_series(x, "sintetica", name) for name, x in cases.items()])


def format_p(p, bound):
    return f"{bound} {p:.2f}" if bound != "=" else f"{p:.3g}"


def report(table):
    view = pd.DataFrame(
        {
            "variable": table.variable,
            "serie": table.serie,
            "ADF p": table.adf_p.map(lambda p: f"{p:.2g}"),
            "KPSS(c) p": [format_p(p, b) for p, b in zip(table.kpss_c_p, table.kpss_c_p_bound)],
            "KPSS(ct) p": [format_p(p, b) for p, b in zip(table.kpss_ct_p, table.kpss_ct_p_bound)],
            "verdetto": table.verdetto,
        }
    )
    return view.to_string(index=False)


def run(verbose=True):
    series = load_series()
    table = pd.DataFrame([test_series(series[v], v, "originale") for v in SERIES])
    lags = kpss_lag_sensitivity(series)
    synthetic = synthetic_check(len(series))
    save_table(table, "stationarity_raw")
    save_table(lags, "stationarity_kpss_lags")
    save_table(synthetic, "stationarity_synthetic")

    if verbose:
        print(report(table))
        print("\nKPSS(c) al variare dei lag:")
        print(lags.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print("\nControllo su serie sintetiche:")
        print(report(synthetic))
    return table


if __name__ == "__main__":
    setup_style()
    run()
