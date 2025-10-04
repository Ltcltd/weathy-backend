import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def build_global_grids():
    """Pre-compute probability grids for fast lookup"""
    
    df = pd.read_csv("data/processed/historical_probabilities.csv")
    
    # Debug: Print available columns
    print("Available columns in data:")
    print(df.columns.tolist())
    print()
    
    output_dir = Path("data/processed/global_grids")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Grid resolution (based on data spacing)
    grid_shape = (36, 72)  # 5° resolution: 180/5 x 360/5
    
    # FIXED: Use correct column names with aggregation suffix
    variables = {
        'rain_occurred': 'rain_occurred_mean',
        'hot_day': 'hot_day_mean',
        'cold_day': 'cold_day_mean',
        'windy_day': 'windy_day_mean',
        'temp': 'temp_mean'
    }
    
    print("Building global probability grids...")
    
    for var_name, col_name in variables.items():
        print(f"\nProcessing {var_name}...")
        
        # Check if column exists
        if col_name not in df.columns:
            print(f"  ⚠ Column '{col_name}' not found, skipping")
            continue
        
        h5_file = output_dir / f"{var_name}_grid.h5"
        
        with h5py.File(h5_file, 'w') as f:
            for day in tqdm(range(1, 367), desc=f"  {var_name}"):
                grid = np.zeros(grid_shape)
                
                day_data = df[df['day_of_year'] == day]
                
                for _, row in day_data.iterrows():
                    # Convert lat/lon to grid indices
                    lat_idx = int((row['lat'] + 90) / 5)
                    lon_idx = int((row['lon'] + 180) / 5)
                    
                    if 0 <= lat_idx < 36 and 0 <= lon_idx < 72:
                        grid[lat_idx, lon_idx] = row[col_name]
                
                # Save with compression
                f.create_dataset(f'day_{day:03d}', data=grid, compression='gzip', compression_opts=4)
        
        print(f"  ✓ {var_name} grid saved")
    
    print(f"\n✓ All grids built: {output_dir}")

if __name__ == "__main__":
    build_global_grids()
