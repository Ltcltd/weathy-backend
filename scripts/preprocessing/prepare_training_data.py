import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def prepare_training_data():
    """
    Convert NASA POWER JSON to training format with ALL 11 weather variables
    
    Uses 10 NASA POWER parameters:
    - T2M, T2M_MAX, T2M_MIN (temperature)
    - PRECTOTCORR (precipitation)
    - WS2M, WS2M_MAX (wind)
    - RH2M (humidity)
    - CLOUD_AMT (cloud cover)
    - ALLSKY_SFC_SW_DWN (solar radiation)
    - PS (pressure)
    """
    data_dir = Path("data/raw/nasa_power")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    files = list(data_dir.glob("*.json"))
    
    print("=" * 80)
    print("PREPARING TRAINING DATA - ALL 11 WEATHER VARIABLES")
    print("=" * 80)
    print(f"\nProcessing {len(files):,} NASA POWER files...")
    
    files_processed = 0
    files_failed = 0
    
    for json_file in tqdm(files, desc="Processing files", ncols=100):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            if 'properties' not in data:
                files_failed += 1
                continue
            
            params = data['properties']['parameter']
            
            # Extract metadata from filename: data_LAT_LON_YEAR.json
            parts = json_file.stem.split('_')
            lat = int(parts[1])
            lon = int(parts[2])
            year = int(parts[3])
            
            # Convert to daily records
            dates = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')
            
            # Extract ALL 10 parameters (with fallbacks)
            temp_vals = list(params.get('T2M', {}).values())
            temp_max_vals = list(params.get('T2M_MAX', {}).values())
            temp_min_vals = list(params.get('T2M_MIN', {}).values())
            precip_vals = list(params.get('PRECTOTCORR', {}).values())
            wind_vals = list(params.get('WS2M', {}).values())
            wind_max_vals = list(params.get('WS2M_MAX', {}).values())
            humid_vals = list(params.get('RH2M', {}).values())
            cloud_vals = list(params.get('CLOUD_AMT', {}).values())
            solar_vals = list(params.get('ALLSKY_SFC_SW_DWN', {}).values())
            pressure_vals = list(params.get('PS', {}).values())
            
            # Use mean temp as fallback for max/min if missing
            if not temp_max_vals:
                temp_max_vals = [t + 5 for t in temp_vals] if temp_vals else []
            if not temp_min_vals:
                temp_min_vals = [t - 5 for t in temp_vals] if temp_vals else []
            if not wind_max_vals:
                wind_max_vals = [w * 1.5 for w in wind_vals] if wind_vals else []
            if not cloud_vals:
                # Infer from solar: low solar = high clouds
                cloud_vals = [max(0, 100 - s/5) for s in solar_vals] if solar_vals else [50] * len(dates)
            
            for i, date in enumerate(dates):
                if i < len(temp_vals):
                    # Get all base measurements
                    temp = temp_vals[i]
                    temp_max = temp_max_vals[i] if i < len(temp_max_vals) else temp + 5
                    temp_min = temp_min_vals[i] if i < len(temp_min_vals) else temp - 5
                    precip = precip_vals[i]
                    wind = wind_vals[i]
                    wind_max = wind_max_vals[i] if i < len(wind_max_vals) else wind * 1.5
                    humidity = humid_vals[i]
                    cloud = cloud_vals[i] if i < len(cloud_vals) else 50
                    solar = solar_vals[i] if i < len(solar_vals) else 200
                    pressure = pressure_vals[i] if i < len(pressure_vals) else 101.3
                    
                    # Determine season for snow detection
                    month = date.month
                    is_winter = (month in [12, 1, 2] and lat > 0) or (month in [6, 7, 8] and lat < 0)
                    
                    # CREATE 11 BINARY LABELS
                    record = {
                        # Location and time
                        'lat': lat,
                        'lon': lon,
                        'date': date,
                        'day_of_year': date.dayofyear,
                        'month': month,
                        
                        # Base measurements (for features)
                        'temp': temp,
                        'temp_max': temp_max,
                        'temp_min': temp_min,
                        'precip': precip,
                        'wind': wind,
                        'wind_max': wind_max,
                        'humidity': humidity,
                        'cloud': cloud,
                        'solar': solar,
                        'pressure': pressure,
                        
                        # === 11 BINARY LABELS ===
                        
                        # 1. RAIN - any significant precipitation
                        'rain': 1 if precip > 1.0 else 0,
                        
                        # 2. HEAVY_RAIN - intense rainfall
                        'heavy_rain': 1 if precip > 10.0 else 0,
                        
                        # 3. SNOW - precipitation when cold (winter only)
                        'snow': 1 if (precip > 1.0 and temp < 2.0 and is_winter) else 0,
                        
                        # 4. CLOUD_COVER_HIGH - significant cloud cover
                        'cloud_cover_high': 1 if cloud > 70 else 0,
                        
                        # 5. WIND_SPEED_HIGH - strong winds
                        'wind_speed_high': 1 if wind_max > 8.0 else 0,
                        
                        # 6. TEMPERATURE_HOT - hot day
                        'temperature_hot': 1 if temp_max > 30 else 0,
                        
                        # 7. TEMPERATURE_COLD - cold day
                        'temperature_cold': 1 if temp_min < 5 else 0,
                        
                        # 8. HEAT_WAVE - extreme heat
                        'heat_wave': 1 if temp_max > 35 else 0,
                        
                        # 9. COLD_SNAP - extreme cold
                        'cold_snap': 1 if temp_min < 0 else 0,
                        
                        # 10. DUST_EVENT - high wind + dry conditions
                        'dust_event': 1 if (wind_max > 10 and precip < 0.1 and humidity < 30) else 0,
                        
                        # 11. UNCOMFORTABLE_INDEX - heat+humidity OR cold+wind
                        'uncomfortable_index': 1 if (
                            (temp_max > 30 and humidity > 70) or  # Hot & humid
                            (temp_min < 5 and wind_max > 8)        # Cold & windy
                        ) else 0
                    }
                    
                    all_records.append(record)
            
            files_processed += 1
            
        except Exception as e:
            files_failed += 1
            continue
    
    # Create DataFrame
    print(f"\nCreating DataFrame...")
    df = pd.DataFrame(all_records)
    
    print(f"\nCalculating historical probabilities per location-day...")
    
    # Calculate probabilities per location and day of year
    prob_df = df.groupby(['lat', 'lon', 'day_of_year']).agg({
        # 11 weather variables
        'rain': 'mean',
        'heavy_rain': 'mean',
        'snow': 'mean',
        'cloud_cover_high': 'mean',
        'wind_speed_high': 'mean',
        'temperature_hot': 'mean',
        'temperature_cold': 'mean',
        'heat_wave': 'mean',
        'cold_snap': 'mean',
        'dust_event': 'mean',
        'uncomfortable_index': 'mean',
        
        # Base measurements (for features)
        'temp': ['mean', 'std'],
        'precip': 'mean',
        'wind': 'mean',
        'humidity': 'mean',
        'cloud': 'mean'
    }).reset_index()
    
    # Flatten column names
    prob_df.columns = ['_'.join(col).strip('_') for col in prob_df.columns.values]
    
    # Save both files
    print(f"\nSaving processed data...")
    df.to_csv(output_dir / 'historical_data.csv', index=False)
    prob_df.to_csv(output_dir / 'historical_probabilities.csv', index=False)
    
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nProcessing Summary:")
    print(f"   • Files processed: {files_processed:,}/{len(files):,}")
    print(f"   • Files failed: {files_failed:,}")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Location-day combinations: {len(prob_df):,}")
    print(f"   • Years of data: ~{len(df) / (365 * len(prob_df['lat'].unique())):.1f}")
    
    print(f"\nALL 11 WEATHER VARIABLES CREATED:")
    labels = ['rain', 'heavy_rain', 'snow', 'cloud_cover_high', 'wind_speed_high',
              'temperature_hot', 'temperature_cold', 'heat_wave', 'cold_snap',
              'dust_event', 'uncomfortable_index']
    
    for i, var in enumerate(labels, 1):
        prob = df[var].mean() * 100
        count = df[var].sum()
        print(f"   {i:2d}. {var:25s}: {prob:5.2f}% ({count:,} occurrences)")
    
    print(f"\nOutput Files:")
    print(f"   • historical_data.csv: {len(df):,} rows × {len(df.columns)} columns")
    print(f"   • historical_probabilities.csv: {len(prob_df):,} rows × {len(prob_df.columns)} columns")
    
    print("\nReady for model training!")
    print("=" * 80)
    
    return df, prob_df


if __name__ == "__main__":
    prepare_training_data()
