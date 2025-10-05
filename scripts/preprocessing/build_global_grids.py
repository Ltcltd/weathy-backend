import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def build_global_grids():
    """
    Pre-compute probability grids for fast lookup
    Creates HDF5 grids for all 11 weather variables + base features
    """
    
    print("=" * 80)
    print("BUILDING GLOBAL PROBABILITY GRIDS - ALL 11 VARIABLES")
    print("=" * 80)
    
    df = pd.read_csv("data/processed/historical_probabilities.csv")
    
    # Debug: Print available columns
    print(f"\nAvailable columns ({len(df.columns)}):")
    print(df.columns.tolist())
    print()
    
    output_dir = Path("data/processed/global_grids")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine grid resolution based on actual data spacing
    unique_lats = sorted(df['lat'].unique())
    unique_lons = sorted(df['lon'].unique())
    
    print(f"Data Coverage:")
    print(f"   Unique latitudes: {len(unique_lats)}")
    print(f"   Unique longitudes: {len(unique_lons)}")
    print(f"   Lat range: {min(unique_lats):.1f}° to {max(unique_lats):.1f}°")
    print(f"   Lon range: {min(unique_lons):.1f}° to {max(unique_lons):.1f}°")
    
    # Calculate spacing safely
    if len(unique_lats) > 1:
        lat_diffs = np.diff(unique_lats)
        lat_spacing = int(np.median(lat_diffs))  # Use median for robustness
    else:
        lat_spacing = 10  # Default fallback
    
    if len(unique_lons) > 1:
        lon_diffs = np.diff(unique_lons)
        lon_spacing = int(np.median(lon_diffs))  # Use median for robustness
    else:
        lon_spacing = 10  # Default fallback
    
    # Ensure spacing is at least 1 degree
    lat_spacing = max(1, lat_spacing)
    lon_spacing = max(1, lon_spacing)
    
    # Grid shape calculation
    n_lats = int(180 / lat_spacing)  # -90 to 90
    n_lons = int(360 / lon_spacing)  # -180 to 180
    grid_shape = (n_lats, n_lons)
    
    print(f"\nGrid Configuration:")
    print(f"   Detected spacing: {lat_spacing}° lat × {lon_spacing}° lon")
    print(f"   Grid shape: {grid_shape[0]} lats × {grid_shape[1]} lons")
    print(f"   Total cells: {grid_shape[0] * grid_shape[1]:,}")
    
    # 11 VARIABLES + base features
    variables = {
        # 11 weather condition probabilities
        'rain': 'rain_mean',
        'heavy_rain': 'heavy_rain_mean',
        'snow': 'snow_mean',
        'cloud_cover_high': 'cloud_cover_high_mean',
        'wind_speed_high': 'wind_speed_high_mean',
        'temperature_hot': 'temperature_hot_mean',
        'temperature_cold': 'temperature_cold_mean',
        'heat_wave': 'heat_wave_mean',
        'cold_snap': 'cold_snap_mean',
        'dust_event': 'dust_event_mean',
        'uncomfortable_index': 'uncomfortable_index_mean',
        
        # Base features (for model inputs)
        'temp': 'temp_mean',
        'temp_std': 'temp_std',
        'precip': 'precip_mean',
        'wind': 'wind_mean',
        'humidity': 'humidity_mean',
        'cloud': 'cloud_mean'
    }
    
    print(f"\nBuilding grids for {len(variables)} variables...")
    print(f"   • 11 weather probabilities")
    print(f"   • 7 base features\n")
    
    grids_built = 0
    grids_skipped = 0
    
    for var_name, col_name in variables.items():
        print(f"\n[{grids_built + grids_skipped + 1}/{len(variables)}] {var_name}...")
        
        # Check if column exists
        if col_name not in df.columns:
            print(f"   ⚠️  Column '{col_name}' not found, skipping")
            grids_skipped += 1
            continue
        
        h5_file = output_dir / f"{var_name}_grid.h5"
        
        try:
            with h5py.File(h5_file, 'w') as f:
                # Add metadata
                f.attrs['variable'] = var_name
                f.attrs['column'] = col_name
                f.attrs['grid_shape'] = grid_shape
                f.attrs['lat_spacing'] = lat_spacing
                f.attrs['lon_spacing'] = lon_spacing
                f.attrs['lat_range'] = [-90, 90]
                f.attrs['lon_range'] = [-180, 180]
                
                # Build grid for each day of year
                for day in tqdm(range(1, 367), desc=f"   Processing", ncols=100, leave=False):
                    grid = np.full(grid_shape, np.nan)  # Use NaN for missing data
                    
                    day_data = df[df['day_of_year'] == day]
                    
                    if len(day_data) == 0:
                        continue
                    
                    for _, row in day_data.iterrows():
                        # Convert lat/lon to grid indices
                        lat_idx = int((row['lat'] + 90) / lat_spacing)
                        lon_idx = int((row['lon'] + 180) / lon_spacing)
                        
                        # Bounds checking
                        if 0 <= lat_idx < grid_shape[0] and 0 <= lon_idx < grid_shape[1]:
                            grid[lat_idx, lon_idx] = row[col_name]
                    
                    # Save with compression
                    f.create_dataset(
                        f'day_{day:03d}',
                        data=grid,
                        compression='gzip',
                        compression_opts=4
                    )
                
                # Calculate and save statistics
                all_values = df[col_name].dropna()
                if len(all_values) > 0:
                    f.attrs['mean'] = float(all_values.mean())
                    f.attrs['std'] = float(all_values.std())
                    f.attrs['min'] = float(all_values.min())
                    f.attrs['max'] = float(all_values.max())
                else:
                    f.attrs['mean'] = 0.0
                    f.attrs['std'] = 0.0
                    f.attrs['min'] = 0.0
                    f.attrs['max'] = 0.0
                
            print(f"Saved {var_name}_grid.h5")
            grids_built += 1
            
        except Exception as e:
            print(f"Failed: {e}")
            grids_skipped += 1
            continue
    
    print("\n" + "=" * 80)
    print("GRID BUILDING COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"   • Grids built: {grids_built}/{len(variables)}")
    print(f"   • Grids skipped: {grids_skipped}")
    print(f"   • Grid resolution: {lat_spacing}° × {lon_spacing}°")
    print(f"   • Days per variable: 366")
    print(f"   • Total files: {grids_built}")
    
    print(f"\nOutput Directory:")
    print(f"   {output_dir}")
    
    # Calculate total file size
    if output_dir.exists():
        h5_files = list(output_dir.glob("*.h5"))
        if h5_files:
            total_size = sum(f.stat().st_size for f in h5_files)
            print(f"\nTotal size: {total_size / (1024**2):.1f} MB")
    
    print(f"\nUsage:")
    print(f"   These grids enable fast spatial interpolation")
    print(f"   Load with: h5py.File('path/to/grid.h5', 'r')")
    print(f"   Access day: grid['day_001'][:]")
    
    print("=" * 80)


if __name__ == "__main__":
    build_global_grids()
