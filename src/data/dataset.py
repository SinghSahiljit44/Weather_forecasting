import xarray
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
NETCDF_PATH = DATA_DIR / "raw" / "era5_land_consolidated_italy.nc"
HOURLY_PATH = DATA_DIR / "interim" / "foggia_hourly.parquet"

# Climatologia fissa in ERA5-Land: si ripete uguale ogni anno, non porta informazione
DROPPED_VARS = ["lai_hv", "lai_lv"]


def load_dataset(path=NETCDF_PATH):
    ds = xarray.open_dataset(path)
    ds = ds.drop_vars(DROPPED_VARS)
    return ds


def select_cell(ds, lat=41.5, lon=15.5):
    ds = ds.sel(latitude=lat, longitude=lon, method="nearest")
    ds = ds.load()
    ds = ds.drop_vars(["latitude", "longitude"])
    return ds


def to_hourly_frame(ds):
    return ds.to_dataframe().sort_index()


def save_hourly(df, path=HOURLY_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def load_hourly(path=HOURLY_PATH):
    return pd.read_parquet(path).sort_index()
