import pandas as pd
import numpy as np

class AnalogMatcher:
    def __init__(self):
        self.prob_data = pd.read_csv('data/processed/historical_probabilities.csv')
    
    def predict(self, lat, lon, day_of_year, variable='rain'):
        """Historical analog matching"""
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
            var_map = {
                'rain': 'rain_occurred_mean',
                'hot': 'hot_day_mean',
                'cold': 'cold_day_mean',
                'windy': 'windy_day_mean'
            }
            col = var_map.get(variable, 'rain_occurred_mean')
            return float(matches.iloc[0][col])
        
        return 0.3  # Default fallback
