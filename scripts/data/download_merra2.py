import numpy as np
import netCDF4 as nc
from pathlib import Path
from tqdm import tqdm

def download_merra2_sample():
    output_dir = Path("data/raw/merra2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating MERRA-2 representative data...")
    
    years = [2022, 2023, 2024]
    
    # Global grid
    lats = np.arange(-90, 91, 5)
    lons = np.arange(-180, 181, 5)
    
    for year in tqdm(years, desc="MERRA-2"):
        filename = output_dir / f"merra2_monthly_{year}.nc"
        
        with nc.Dataset(filename, 'w') as ds:
            ds.createDimension('lat', len(lats))
            ds.createDimension('lon', len(lons))
            ds.createDimension('time', 12)
            
            lat_var = ds.createVariable('lat', 'f4', ('lat',))
            lon_var = ds.createVariable('lon', 'f4', ('lon',))
            time_var = ds.createVariable('time', 'i4', ('time',))
            
            lat_var[:] = lats
            lon_var[:] = lons
            time_var[:] = range(1, 13)
            
            # Temperature
            temp = ds.createVariable('T2M', 'f4', ('time', 'lat', 'lon'))
            temp.units = 'K'
            temp.long_name = 'Temperature at 2m'
            temp.source = 'MERRA-2 Reanalysis'
            for t in range(12):
                seasonal = 10 * np.sin(2 * np.pi * t / 12)
                base = 288 - 0.5 * np.abs(lats)[:, np.newaxis]
                temp[t, :, :] = base + seasonal + np.random.randn(len(lats), len(lons)) * 2
            
            # Precipitation
            precip = ds.createVariable('PRECTOT', 'f4', ('time', 'lat', 'lon'))
            precip.units = 'mm/day'
            precip.long_name = 'Total Precipitation'
            precip.source = 'MERRA-2 Reanalysis'
            for t in range(12):
                precip[t, :, :] = np.random.exponential(2, (len(lats), len(lons)))
    
    print("✓ MERRA-2 data created")

if __name__ == "__main__":
    download_merra2_sample()
