#!/usr/bin/env python3
import subprocess
import sys

scripts = [
    ("scripts/data/download_nasa_power.py", "NASA POWER"),
    ("scripts/data/download_climate_indices.py", "Climate Indices"),
    ("scripts/data/download_merra2.py", "MERRA-2"),
    ("scripts/data/download_giovanni.py", "Giovanni"),
    ("scripts/data/download_satellite.py", "Satellite Docs"),
    ("scripts/data/download_era5.py", "ERA5"),
]

print("="*80)
print("DOWNLOADING MULTIPLE NASA DATA SOURCES")
print("="*80)
print()

for i, (script, desc) in enumerate(scripts, 1):
    print(f"[{i}/{len(scripts)}] {desc}")
    subprocess.run([sys.executable, script])
    print()

print("="*80)
print("✓ ALL DATA SOURCES DOWNLOADED")
print("="*80)
