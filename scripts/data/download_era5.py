import numpy as np
import netCDF4 as nc
from pathlib import Path
from tqdm import tqdm

def download_era5_sample():
    """
    ERA5 representative data
    Real ERA5 requires CDS API registration
    For hackathon: Create representative structure
    """
    
    output_dir = Path("data/raw/era5")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating ERA5 representative data...")
    print("Note: Production would use Copernicus CDS API")
    
    years = [2022, 2023, 2024]
    
    # ERA5 grid (0.25° resolution, simplified to 1° for speed)
    lats = np.arange(-90, 90.1, 1.0)
    lons = np.arange(-180, 180.1, 1.0)
    
    for year in tqdm(years, desc="ERA5"):
        filename = output_dir / f"era5_monthly_{year}.nc"
        
        with nc.Dataset(filename, 'w') as ds:
            # Dimensions
            ds.createDimension('latitude', len(lats))
            ds.createDimension('longitude', len(lons))
            ds.createDimension('time', 12)
            
            # Coordinate variables
            lat_var = ds.createVariable('latitude', 'f4', ('latitude',))
            lon_var = ds.createVariable('longitude', 'f4', ('longitude',))
            time_var = ds.createVariable('time', 'i4', ('time',))
            
            lat_var[:] = lats
            lon_var[:] = lons
            time_var[:] = range(1, 13)
            
            lat_var.units = 'degrees_north'
            lon_var.units = 'degrees_east'
            time_var.units = 'month'
            
            # Variables
            # 2m Temperature
            t2m = ds.createVariable('t2m', 'f4', ('time', 'latitude', 'longitude'))
            t2m.units = 'K'
            t2m.long_name = '2 metre temperature'
            t2m.source = 'ERA5 Reanalysis'
            
            # Total Precipitation
            tp = ds.createVariable('tp', 'f4', ('time', 'latitude', 'longitude'))
            tp.units = 'm'
            tp.long_name = 'Total precipitation'
            tp.source = 'ERA5 Reanalysis'
            
            # 10m Wind U/V components
            u10 = ds.createVariable('u10', 'f4', ('time', 'latitude', 'longitude'))
            u10.units = 'm/s'
            u10.long_name = '10 metre U wind component'
            
            v10 = ds.createVariable('v10', 'f4', ('time', 'latitude', 'longitude'))
            v10.units = 'm/s'
            v10.long_name = '10 metre V wind component'
            
            # Fill with realistic data
            for t in range(12):
                # Temperature with seasonal cycle
                seasonal = 10 * np.sin(2 * np.pi * t / 12)
                lat_gradient = 288 - 0.5 * np.abs(lats)[:, np.newaxis]
                t2m[t, :, :] = lat_gradient + seasonal + np.random.randn(len(lats), len(lons)) * 3
                
                # Precipitation
                tp[t, :, :] = np.random.exponential(0.003, (len(lats), len(lons)))
                
                # Wind components
                u10[t, :, :] = np.random.randn(len(lats), len(lons)) * 5
                v10[t, :, :] = np.random.randn(len(lats), len(lons)) * 5
    
    print("✓ ERA5 data created")
    print("  Production: Use 'cdsapi' library with Copernicus account")

if __name__ == "__main__":
    download_era5_sample()
