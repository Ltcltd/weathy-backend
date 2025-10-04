import requests
import json
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import time

def download_single_location(args):
    """Download data for a single location-year"""
    lat, lon, year = args
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    output_dir = Path("data/raw/nasa_power")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"data_{lat}_{lon}_{year}.json"
    
    # Skip if already downloaded
    if filename.exists():
        return True
    
    params = {
        "parameters": "T2M,PRECTOTCORR,WS2M,RH2M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON"
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=30)
            if response.status_code == 200:
                with open(filename, 'w') as f:
                    json.dump(response.json(), f)
                return True
            time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return False
            time.sleep(2)
    
    return False

def download_nasa_power_fast():
    """
    Download NASA POWER data with parallel processing
    Strategic sampling for hackathon - scalable to full grid
    """
    
    # Core sample (50 locations) - covers major climate zones
    # for hackathon, scales to full grid by reducing step size
    
    # Sample every 15° latitude, 30° longitude (50 locations total)
    lats = list(range(-75, 90, 15))  # -75 to 75, every 15°
    lons = list(range(-165, 180, 30))  # -165 to 165, every 30°
    
    # Last 3 years only for speed (scales to more years easily)
    years = [2022, 2023, 2024]
    
    # Create all download tasks
    tasks = [(lat, lon, year) for lat in lats for lon in lons for year in years]
    
    print(f"Downloading {len(tasks)} location-years from NASA POWER API...")
    print(f"Locations: {len(lats) * len(lons)}, Years: {len(years)}")
    print(f"Resolution: ~15° lat × 30° lon (scalable to any resolution)")
    print(f"Estimated time: 10-15 minutes\n")
    
    # Parallel download with 15 workers (balance speed vs API limits)
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        results = list(tqdm(
            executor.map(download_single_location, tasks),
            total=len(tasks),
            desc="Downloading",
            unit="files"
        ))
    
    successful = sum(results)
    failed = len(results) - successful
    
    print(f"\n✓ Downloaded: {successful}/{len(tasks)} files")
    if failed > 0:
        print(f"⚠ Failed: {failed} files (will use interpolation)")
    
    print(f"\n To scale to full global coverage:")
    print(f"   Change: range(-75, 90, 15) → range(-90, 95, 5)  [5° resolution]")
    print(f"   Change: range(-165, 180, 30) → range(-180, 185, 5)  [5° resolution]")
    print(f"   This gives ~2,500 locations instead of 50")

if __name__ == "__main__":
    download_nasa_power_fast()
