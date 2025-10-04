import json
from pathlib import Path

def download_giovanni_metadata():
    output_dir = Path("data/raw/giovanni")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating Giovanni metadata...")
    
    regions = [
        {"name": "North_America", "lat": [25, 50], "lon": [-125, -70]},
        {"name": "Europe", "lat": [35, 60], "lon": [-10, 40]},
        {"name": "Asia", "lat": [20, 50], "lon": [70, 140]},
        {"name": "Australia", "lat": [-40, -10], "lon": [110, 155]}
    ]
    
    years = [2022, 2023, 2024]
    
    for region in regions:
        for year in years:
            metadata = {
                'source': 'NASA Giovanni',
                'dataset': 'TRMM_3B42_Daily',
                'region': region['name'],
                'year': year,
                'lat_range': region['lat'],
                'lon_range': region['lon'],
                'variable': 'precipitation',
                'url': 'https://giovanni.gsfc.nasa.gov/giovanni/'
            }
            
            filename = output_dir / f"giovanni_{region['name']}_{year}.json"
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    print("✓ Giovanni metadata created")

if __name__ == "__main__":
    download_giovanni_metadata()
