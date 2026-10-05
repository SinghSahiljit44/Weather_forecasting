import pandas as pd
from pathlib import Path

from src.data.dataset import DATA_DIR

DAILY_PATH = DATA_DIR / "processed" / "foggia_daily.parquet"

TEMP_VARS = ["t2m", "skt", "stl1", "stl2", "stl3", "stl4"]

SOIL_MOIST_VARS = ["swvl1", "swvl2", "swvl3", "swvl4"]

INSTANT_VARS = TEMP_VARS + SOIL_MOIST_VARS

ACCUM_VARS = ["tp", "e", "pev"]

COLUMNS = (
    ["t2m_mean", "t2m_min", "t2m_max"]
    + [f"{v}_mm" for v in ACCUM_VARS]
    + [f"{v}_mean" for v in INSTANT_VARS if v != "t2m"]
)


def _daily_instantaneous(df):
    daily = df[INSTANT_VARS].resample("D").mean()
    daily = daily.rename(columns={v: f"{v}_mean" for v in INSTANT_VARS})
    daily["t2m_min"] = df["t2m"].resample("D").min()
    daily["t2m_max"] = df["t2m"].resample("D").max()
    return daily


def _daily_accumulated(df):
    midnight = df.loc[df.index.hour == 0, ACCUM_VARS]
    daily = midnight.set_axis(midnight.index.normalize() - pd.Timedelta(days=1))
    return daily.rename(columns={v: f"{v}_mm" for v in ACCUM_VARS})


def hourly_to_daily(df):
    df = df.sort_index()
    daily = _daily_instantaneous(df).join(_daily_accumulated(df), how="inner")
    daily = daily.dropna() # last day is dropped, also first day
    daily.index.name = "date"
    return daily[COLUMNS]


def save_daily(df, path=DAILY_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_daily(path=DAILY_PATH):
    return pd.read_parquet(path)