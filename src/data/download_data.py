import argparse
import os
import time
import cdsapi

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

AREA = [41.8, 15.0, 41.1, 16.0]  # Bounding box Foggia / Capitanata (N, W, S, E)
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]  # le date inesistenti (es. 30 febbraio) vengono ignorate dal CDS

# Il dataset delle statistiche giornaliere ha un limite di 400 "giorni-variabile" per richiesta:
# un anno intero ci sta solo chiedendo una variabile alla volta.
DAILY_MEAN_VARIABLES = ["2m_temperature", "surface_pressure"]

POLL_SECONDS = 30


def daily_mean_requests(years, output_dir):
    for year in years:
        for variable in DAILY_MEAN_VARIABLES:
            params = {
                "variable": [variable],
                "year": year,
                "month": MONTHS,
                "day": DAYS,
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "1_hourly",
                "area": AREA,
            }
            # Con una sola variabile il CDS restituisce direttamente il NetCDF, non uno zip
            path = os.path.join(output_dir, f"era5land_{variable}_{year}.nc")
            yield f"{variable} {year}", "derived-era5-land-daily-statistics", params, path


def precipitation_requests(years, output_dir):
    # total_precipitation non esiste nelle statistiche giornaliere: la prendo dal dataset orario.
    # In ERA5-Land è cumulata a partire dalle 00 UTC, quindi il valore delle 00:00 del giorno D+1
    # è il totale del giorno D. Basta un orario al giorno, più il 1° gennaio dell'anno successivo
    # per avere il totale dell'ultimo 31 dicembre.
    params = {
        "variable": ["total_precipitation"],
        "year": years,
        "month": MONTHS,
        "day": DAYS,
        "time": ["00:00"],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    path = os.path.join(output_dir, f"era5land_total_precipitation_{years[0]}-{years[-1]}.nc")
    yield f"total_precipitation {years[0]}-{years[-1]}", "reanalysis-era5-land", params, path

    next_year = str(int(years[-1]) + 1)
    params = {**params, "year": [next_year], "month": ["01"], "day": ["01"]}
    path = os.path.join(output_dir, f"era5land_total_precipitation_{next_year}-01-01.nc")
    yield f"total_precipitation {next_year}-01-01", "reanalysis-era5-land", params, path


def download_foggia_data(start_year=2020, end_year=2025, output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    # wait_until_complete=False: retrieve() invia la richiesta e ritorna subito, così tutte
    # le richieste stanno in coda sul CDS in parallelo invece di aspettarsi a vicenda
    client = cdsapi.Client(wait_until_complete=False, quiet=True)
    years = [str(y) for y in range(start_year, end_year + 1)]
    requests = [*precipitation_requests(years, output_dir), *daily_mean_requests(years, output_dir)]

    pending = {}
    for label, dataset, params, path in requests:
        if os.path.exists(path):
            print(f"{label} già scaricato. Salto...")
            continue
        try:
            pending[label] = (client.retrieve(dataset, params), path)
            print(f"{label}: richiesta inviata")
        except Exception as e:
            print(f"Errore invio {label}: {e}")

    total = len(pending)
    done = 0
    last_status = {}
    while pending:
        for label, (remote, path) in list(pending.items()):
            try:
                ready = remote.results_ready  # solleva un'eccezione se la richiesta è fallita
            except Exception as e:
                print(f"Errore {label}: {e}")
                del pending[label]
                continue

            if not ready:
                status = remote.last_status
                if last_status.get(label) != status:
                    print(f"{label}: {status}")
                    last_status[label] = status
                continue

            # Scarico su un file temporaneo e rinomino solo a download completato,
            # così un'interruzione non lascia un file incompleto che verrebbe poi saltato
            tmp_path = path + ".part"
            try:
                remote.download(tmp_path)
                os.replace(tmp_path, path)
                done += 1
                print(f"[{done}/{total}] {label}: completato -> {path}")
            except Exception as e:
                print(f"Errore download {label}: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            del pending[label]

        if pending:
            time.sleep(POLL_SECONDS)

    print(f"Fatto: {done}/{total} file scaricati.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scarica i dati ERA5-Land per l'area di Foggia dal CDS")
    parser.add_argument("--start", type=int, default=2020, help="primo anno (incluso)")
    parser.add_argument("--end", type=int, default=2025, help="ultimo anno (incluso)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    download_foggia_data(args.start, args.end, args.output_dir)
