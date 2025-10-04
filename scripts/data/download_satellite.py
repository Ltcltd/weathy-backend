import json
from pathlib import Path

def document_satellite_sources():
    output_dir = Path("data/raw/satellite")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Documenting satellite data sources...")
    
    sources = {
        "GOES-16": {
            "description": "Geostationary weather satellite",
            "variables": ["cloud_top_temperature", "cloud_optical_depth", "AOD"],
            "url": "https://www.goes.noaa.gov/",
            "usage": "Real-time cloud imagery for precipitation analysis",
            "spatial_resolution": "2-4 km",
            "temporal_resolution": "15 minutes"
        },
        "MODIS": {
            "description": "Moderate Resolution Imaging Spectroradiometer",
            "variables": ["LST_Day", "LST_Night", "NDVI", "EVI"],
            "url": "https://modis.gsfc.nasa.gov/",
            "usage": "Land surface temperature and vegetation monitoring",
            "spatial_resolution": "1 km",
            "temporal_resolution": "Daily"
        },
        "GPM": {
            "description": "Global Precipitation Measurement",
            "variables": ["precipitation_rate", "precipitation_probability"],
            "url": "https://gpm.nasa.gov/",
            "usage": "High-resolution global precipitation",
            "spatial_resolution": "0.1°",
            "temporal_resolution": "30 minutes"
        }
    }
    
    with open(output_dir / "satellite_sources.json", 'w') as f:
        json.dump(sources, f, indent=2)
    
    print("✓ Satellite sources documented")

if __name__ == "__main__":
    document_satellite_sources()
