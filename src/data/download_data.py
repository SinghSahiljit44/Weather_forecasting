import os
import cdsapi

def download_foggia_data(output_dir="./data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    c = cdsapi.Client()
    
    years = [str(y) for y in range(2020, 2026)]
    months = [f"{m:02d}" for m in range(1, 13)]

    total_files = len(years) * len(months)
    current = 0

    for year in years:
        for month in months:
            current += 1
            zip_path = os.path.join(output_dir, f"era5_foggia_{year}_{month}.zip")
            
            if os.path.exists(zip_path):
                print(f"[{current}/{total_files}] {year}-{month} già scaricato. Salto...")
                continue

            params = {
                "variable": [
                    "2m_temperature",
                    "surface_pressure",
                    "total_precipitation"
                ],
                "year": year,
                "month": [month],  # 1 mese alla volta per rispettare i limiti di Copernicus
                "day": [f"{d:02d}" for d in range(1, 32)],
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "area": [41.8, 15.0, 41.1, 16.0],  # Bounding box Foggia / Capitanata
                "format": "netcdf",
                "download_format": "zip"
            }

            print(f"[{current}/{total_files}] Inizio download {year}-{month}...")
            try:
                res = c.retrieve("derived-era5-land-daily-statistics", params)
                res.download(zip_path)
            except Exception as e:
                print(f"Errore download {year}-{month}: {e}")

if __name__ == "__main__":
    download_foggia_data()