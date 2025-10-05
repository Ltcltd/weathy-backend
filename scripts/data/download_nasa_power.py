import requests
import json
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time, random

# Reuse connections
session = requests.Session()

def download_single_location(args):
    """Download data for a single location-year with ALL relevant parameters"""
    lat, lon, year = args
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    output_dir = Path("data/raw/nasa_power")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"data_{lat}_{lon}_{year}.json"
    if filename.exists():
        return True
    
    params = {
        "parameters": ",".join([
            "T2M","T2M_MAX","T2M_MIN",
            "PRECTOTCORR",
            "WS2M","WS2M_MAX",
            "RH2M",
            "CLOUD_AMT","ALLSKY_SFC_SW_DWN",
            "PS"
        ]),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON"
    }
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = session.get(base_url, params=params, timeout=45)
            if response.status_code == 200:
                with open(filename, 'w') as f:
                    json.dump(response.json(), f)
                return True
            elif response.status_code == 429:
                # exponential backoff with jitter
                wait = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
            else:
                time.sleep(1 + random.uniform(0, 0.5))
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed {lat},{lon},{year}: {e}")
                return False
            time.sleep(2 + random.uniform(0, 1))
    
    return False


def download_nasa_power_comprehensive():
    lats = list(range(-80, 85, 10))
    lons = list(range(-175, 180, 10))
    years = [2022, 2023, 2024]
    
    tasks = [(lat, lon, year) for lat in lats for lon in lons for year in years]
    
    print("=" * 80)
    print("NASA POWER DATA DOWNLOAD")
    print("=" * 80)
    print(f"   Locations: {len(lats) * len(lons)}")
    print(f"   Years: {len(years)}")
    print(f"   Total files: {len(tasks)}\n")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(
            executor.map(download_single_location, tasks),
            total=len(tasks),
            desc="Downloading NASA POWER",
            unit="files",
            ncols=100
        ))
    
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"\nDownloaded: {successful}/{len(tasks)} files")
    if failed > 0:
        print(f"Failed: {failed} files")


if __name__ == "__main__":
    download_nasa_power_comprehensive()
