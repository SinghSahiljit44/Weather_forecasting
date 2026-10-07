import json

import numpy as np
import pandas as pd
from scipy import stats

from src.data.dataset import DATA_DIR
from src.data.preprocessing import load_daily

EDA_DIR = DATA_DIR / "eda"
SMOOTH_WINDOW = 31  
ACF_LAGS = list(range(0, 61)) + [91, 182, 365, 730]
RAIN_THRESHOLD = 1.0 


def _mmdd(index):
    return index.strftime("%m-%d")


def _smooth_circular(s, window=SMOOTH_WINDOW):
    half = window // 2
    padded = np.r_[s.values[-half:], s.values, s.values[:half]]
    kernel = np.ones(window) / window
    return pd.Series(np.convolve(padded, kernel, mode="valid"), index=s.index)


def add_anomalies(d):
    out = d.copy()
    key = _mmdd(d.index)
    for col in ["t2m_mean", "tp_mm", "swvl1_mean"]:
        out[f"{col}_anom"] = d[col] - d.groupby(key)[col].transform("mean")
    return out


def acf(x, lags=ACF_LAGS):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    den = (x * x).sum()
    return pd.Series(
        [1.0 if lag == 0 else float((x[:-lag] * x[lag:]).sum() / den) for lag in lags],
        index=pd.Index(lags, name="lag"),
    )


def table_annual(d):
    ann = d.groupby(d.index.year).agg(
        t2m_mean=("t2m_mean", "mean"),
        t2m_max=("t2m_max", "max"),
        t2m_min=("t2m_min", "min"),
        tp_total=("tp_mm", "sum"),
        rainy_days=("tp_mm", lambda s: int((s >= RAIN_THRESHOLD).sum())),
        swvl1_mean=("swvl1_mean", "mean"),
        days=("tp_mm", "size"),
    )
    ann.index.name = "year"
    return ann


def table_monthly_series(d):
    m = d.resample("MS").agg(t2m_mean=("t2m_mean", "mean"), tp_total=("tp_mm", "sum"))
    m.index.name = "month_start"
    return m


def table_monthly_climatology(d):
    n_years = d.index.year.nunique()
    clim = d.groupby(d.index.month).agg(
        t2m_mean=("t2m_mean", "mean"),
        t2m_p10=("t2m_mean", lambda s: s.quantile(0.10)),
        t2m_p90=("t2m_mean", lambda s: s.quantile(0.90)),
        tp_mean_total=("tp_mm", lambda s: s.sum() / n_years),
        rainy_days=("tp_mm", lambda s: (s >= RAIN_THRESHOLD).sum() / n_years),
    )
    clim.index.name = "month"
    return clim


def table_doy_climatology(d):
    key = _mmdd(d.index)
    clim = d.groupby(key).agg(t2m_mean=("t2m_mean", "mean"), tp_mean=("tp_mm", "mean"))
    clim.index.name = "mmdd"
    clim["t2m_smooth"] = _smooth_circular(clim["t2m_mean"])
    clim["tp_smooth"] = _smooth_circular(clim["tp_mean"])
    clim.insert(0, "doy", np.arange(1, len(clim) + 1))
    return clim


def table_acf(d):
    a = add_anomalies(d)
    return pd.DataFrame(
        {
            "t2m_raw": acf(a.t2m_mean),
            "t2m_anom": acf(a.t2m_mean_anom),
            "tp_raw": acf(a.tp_mm),
            "tp_anom": acf(a.tp_mm_anom),
            "swvl1_anom": acf(a.swvl1_mean_anom),
        }
    )


def table_daily_anomalies(d):
    a = add_anomalies(d)
    cols = ["t2m_mean", "t2m_mean_anom", "tp_mm", "tp_mm_anom", "swvl1_mean", "swvl1_mean_anom"]
    out = a[cols].copy()
    out["year"] = a.index.year
    out["doy"] = a.index.dayofyear
    out["t2m_jump"] = a.t2m_mean.diff()
    out["t2m_roll365"] = a.t2m_mean.rolling(365, center=True).mean()
    return out


def table_rain_distribution(d):
    edges = np.array([0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 40, 50, np.inf])
    labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
    binned = pd.cut(d.tp_mm, bins=edges, right=False, labels=labels)
    dist = pd.DataFrame(
        {
            "days": binned.value_counts().reindex(labels),
            "rain_total": d.tp_mm.groupby(binned, observed=False).sum().reindex(labels),
        }
    )
    dist["share_days"] = dist.days / dist.days.sum()
    dist["share_rain"] = dist.rain_total / dist.rain_total.sum()
    dist.index.name = "bin_mm"
    return dist


def table_extremes(d, n=10):
    a = add_anomalies(d)
    rows = []
    for name, series, ascending in [
        ("caldo_anomalo", a.t2m_mean_anom, False),
        ("freddo_anomalo", a.t2m_mean_anom, True),
        ("pioggia_massima", a.tp_mm, False),
        ("salto_termico", a.t2m_mean.diff().abs(), False),
    ]:
        top = series.nsmallest(n) if ascending else series.nlargest(n)
        for date, value in top.items():
            rows.append(
                {
                    "kind": name,
                    "date": date.date(),
                    "value": round(float(value), 2),
                    "t2m_mean": round(float(a.t2m_mean.loc[date]), 2),
                    "tp_mm": round(float(a.tp_mm.loc[date]), 2),
                }
            )
    return pd.DataFrame(rows)


def summary(d):
    a = add_anomalies(d)
    ann = table_annual(d)
    full_years = ann[ann.days >= 365]
    out = {}

    expected = pd.date_range(d.index[0], d.index[-1], freq="D")
    out["rows"] = int(len(d))
    out["first_day"] = str(d.index[0].date())
    out["last_day"] = str(d.index[-1].date())
    out["missing_days"] = int(len(expected.difference(d.index)))
    out["nan_cells"] = int(d.isna().sum().sum())
    out["duplicated_dates"] = int(d.index.duplicated().sum())

    for col, key in [("t2m_mean", "t2m"), ("tp_total", "tp")]:
        slope, _, r, p, _ = stats.linregress(full_years.index, full_years[col])
        out[f"trend_{key}_per_decade"] = round(float(slope) * 10, 4)
        out[f"trend_{key}_pvalue"] = float(f"{p:.2e}")
        out[f"trend_{key}_r2"] = round(float(r) ** 2, 3)
    out["t2m_first_decade"] = round(float(full_years.loc[:1999, "t2m_mean"].mean()), 2)
    out["t2m_last_decade"] = round(float(full_years.loc[2016:, "t2m_mean"].mean()), 2)
    out["tp_annual_mean"] = round(float(full_years.tp_total.mean()), 1)
    out["tp_annual_min"] = [int(full_years.tp_total.idxmin()), round(float(full_years.tp_total.min()), 1)]
    out["tp_annual_max"] = [int(full_years.tp_total.idxmax()), round(float(full_years.tp_total.max()), 1)]

    doy = table_doy_climatology(d)
    out["t2m_seasonal_amplitude"] = round(float(doy.t2m_smooth.max() - doy.t2m_smooth.min()), 2)
    for col, anom, key in [("t2m_mean", "t2m_mean_anom", "t2m"), ("tp_mm", "tp_mm_anom", "tp")]:
        out[f"seasonal_variance_share_{key}"] = round(1 - float(a[anom].var() / a[col].var()), 3)

    acf_tab = table_acf(d)
    for col in acf_tab.columns:
        out[f"acf_{col}"] = {str(lag): round(float(acf_tab[col].loc[lag]), 3) for lag in [1, 3, 7, 30, 365]}

    out["t2m_abs_max"] = [str(d.t2m_max.idxmax().date()), round(float(d.t2m_max.max()), 1)]
    out["t2m_abs_min"] = [str(d.t2m_min.idxmin().date()), round(float(d.t2m_min.min()), 1)]
    z = (a.t2m_mean_anom - a.t2m_mean_anom.mean()) / a.t2m_mean_anom.std()
    out["t2m_anom_outliers_z4"] = int((z.abs() > 4).sum())
    out["t2m_jump_mean"] = round(float(d.t2m_mean.diff().abs().mean()), 2)
    out["t2m_jump_max"] = round(float(d.t2m_mean.diff().abs().max()), 2)

    out["dry_days_exact_zero_share"] = round(float((d.tp_mm == 0).mean()), 3)
    out["dry_days_below_0.1mm_share"] = round(float((d.tp_mm < 0.1).mean()), 3)
    out["rainy_days_share"] = round(float((d.tp_mm >= RAIN_THRESHOLD).mean()), 3)
    out["tp_skewness"] = round(float(d.tp_mm.skew()), 2)
    out["tp_quantiles"] = {q: round(float(d.tp_mm.quantile(q)), 2) for q in [0.5, 0.9, 0.99, 0.999]}
    out["tp_max"] = [str(d.tp_mm.idxmax().date()), round(float(d.tp_mm.max()), 1)]
    top1pct = d.tp_mm.nlargest(int(len(d) * 0.01)).sum()
    out["rain_share_from_wettest_1pct"] = round(float(top1pct / d.tp_mm.sum()), 3)
    out["evaporation_positive_days"] = [int((d.e_mm > 0).sum()), int((d.pev_mm > 0).sum())]
    return out


TABLES = {
    "annual.csv": table_annual,
    "monthly_series.csv": table_monthly_series,
    "monthly_climatology.csv": table_monthly_climatology,
    "doy_climatology.csv": table_doy_climatology,
    "acf.csv": table_acf,
    "daily_anomalies.csv": table_daily_anomalies,
    "rain_distribution.csv": table_rain_distribution,
    "extremes.csv": table_extremes,
}


def build(out_dir=EDA_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_daily()
    for name, fn in TABLES.items():
        table = fn(d)
        table.to_csv(out_dir / name, index=not name == "extremes.csv", float_format="%.4f")
        print(f"{name:26s} {table.shape[0]:6d} righe x {table.shape[1]} colonne")
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary(d), fh, indent=2)
    print(f"{'summary.json':26s} {len(summary(d))} voci")


if __name__ == "__main__":
    build()
