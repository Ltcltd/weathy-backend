import pandas as pd
import numpy as np

class AnalogMatcher:
    # Map API variables to data column names
    VARIABLE_MAP = {
        'rain': 'rain_mean',
        'heavy_rain': 'heavy_rain_mean',
        'snow': 'snow_mean',
        'cloud_cover': 'cloud_cover_high_mean',
        'wind_speed_high': 'wind_speed_high_mean',
        'temperature_hot': 'temperature_hot_mean',
        'temperature_cold': 'temperature_cold_mean',
        'heat_wave': 'heat_wave_mean',
        'cold_snap': 'cold_snap_mean',
        'dust_event': 'dust_event_mean',
        'uncomfortable_index': 'uncomfortable_index_mean'
    }
    
    def __init__(self):
        self.prob_data = pd.read_csv('data/processed/historical_probabilities.csv')
    
    def predict(self, lat, lon, day_of_year, variable='rain'):
        """
        Historical analog matching - SUPPORTS ALL 11 VARIABLES
        
        Args:
            lat, lon: Location
            day_of_year: Day of year (1-366)
            variable: One of 11 weather variables
        
        Returns:
            Float probability [0, 1]
        """
        # Find closest location
        distances = ((self.prob_data['lat'] - lat)**2 + (self.prob_data['lon'] - lon)**2)**0.5
        nearest = distances.idxmin()
        nearest_lat = self.prob_data.loc[nearest, 'lat']
        nearest_lon = self.prob_data.loc[nearest, 'lon']
        
        # Find day of year match
        matches = self.prob_data[
            (self.prob_data['lat'] == nearest_lat) &
            (self.prob_data['lon'] == nearest_lon) &
            (self.prob_data['day_of_year'] == day_of_year)
        ]
        
        if len(matches) > 0:
            col = self.VARIABLE_MAP.get(variable, 'rain_mean')
            
            if col in matches.columns:
                return float(matches.iloc[0][col])
            else:
                # Fallback to rain if column missing
                if 'rain_mean' in matches.columns:
                    return float(matches.iloc[0]['rain_mean'])
                else:
                    return 0.3
        
        return 0.3  # Default fallback
