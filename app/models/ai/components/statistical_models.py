import pickle
import numpy as np

class StatisticalModels:
    def __init__(self):
        with open('models/statistical/models.pkl', 'rb') as f:
            self.models = pickle.load(f)
        with open('models/statistical/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
    
    def predict(self, lat, lon, day_of_year, hist_temp, hist_precip, hist_wind, enso=0, nao=0, pdo=0, variable='rain'):
        """Predict using statistical models with HISTORICAL features
        
        Parameters match training data structure:
        - lat, lon, day_of_year: location and time
        - hist_temp, hist_precip, hist_wind: historical averages
        - enso, nao, pdo: climate indices
        - variable: 'rain', 'hot', 'cold', 'windy', 'temp'
        """
        # Build feature vector matching training EXACTLY
        # Order: [lat, lon, day_of_year, temp_mean, precip_mean, wind_mean, enso, nao, pdo]
        features = np.array([[
            lat, lon, day_of_year,
            hist_temp,    # temp_mean
            hist_precip,  # precip_mean
            hist_wind,    # wind_mean
            enso, nao, pdo  # climate indices
        ]])
        
        # Get models for this variable
        var_models = self.models.get(variable, self.models['rain'])
        
        # Average RF and XGBoost predictions
        rf_pred = var_models['rf'].predict(features)[0]
        xgb_pred = var_models['xgb'].predict(features)[0]
        avg_pred = (rf_pred + xgb_pred) / 2.0
        
        return float(np.clip(avg_pred, 0, 1))
