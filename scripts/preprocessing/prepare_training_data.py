import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare_training_data():
    """Convert NASA POWER JSON to training format"""
    
    data_dir = Path("data/raw/nasa_power")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    files = list(data_dir.glob("*.json"))
    print(f"Processing {len(files)} NASA POWER files...")
    
    for json_file in tqdm(files, desc="Processing"):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            if 'properties' not in data:
                continue
            
            params = data['properties']['parameter']
            
            # Extract metadata from filename: data_LAT_LON_YEAR.json
            parts = json_file.stem.split('_')
            lat = int(parts[1])
            lon = int(parts[2])
            year = int(parts[3])
            
            # Convert to daily records
            dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
            
            temp_vals = list(params.get('T2M', {}).values())
            precip_vals = list(params.get('PRECTOTCORR', {}).values())
            wind_vals = list(params.get('WS2M', {}).values())
            humid_vals = list(params.get('RH2M', {}).values())
            
            for i, date in enumerate(dates):
                if i < len(temp_vals):
                    all_records.append({
                        'lat': lat,
                        'lon': lon,
                        'date': date,
                        'day_of_year': date.dayofyear,
                        'month': date.month,
                        'temp': temp_vals[i],
                        'precip': precip_vals[i],
                        'wind': wind_vals[i],
                        'humidity': humid_vals[i],
                        'rain_occurred': 1 if precip_vals[i] > 1.0 else 0,
                        'hot_day': 1 if temp_vals[i] > 30 else 0,
                        'cold_day': 1 if temp_vals[i] < 5 else 0,
                        'windy_day': 1 if wind_vals[i] > 8 else 0
                    })
        except Exception as e:
            continue
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    print(f"\nCalculating historical probabilities...")
    # Calculate probabilities per location and day of year
    prob_df = df.groupby(['lat', 'lon', 'day_of_year']).agg({
        'rain_occurred': 'mean',
        'hot_day': 'mean',
        'cold_day': 'mean',
        'windy_day': 'mean',
        'temp': ['mean', 'std'],
        'precip': 'mean',
        'wind': 'mean'
    }).reset_index()
    
    prob_df.columns = ['_'.join(col).strip('_') for col in prob_df.columns.values]
    
    # Save
    df.to_csv(output_dir / 'historical_data.csv', index=False)
    prob_df.to_csv(output_dir / 'historical_probabilities.csv', index=False)
    
    print(f"\n✓ Processed {len(df):,} records")
    print(f"✓ Created probabilities for {len(prob_df):,} location-days")
    
    return df, prob_df

if __name__ == "__main__":
    prepare_training_data()
