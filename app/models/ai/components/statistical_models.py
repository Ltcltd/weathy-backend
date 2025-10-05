import pickle
import numpy as np

class StatisticalModels:
    def __init__(self):
        with open('models/statistical/models.pkl', 'rb') as f:
            self.models = pickle.load(f)
        with open('models/statistical/feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
    
    def predict(self, lat, lon, day_of_year, hist_temp, hist_precip, hist_wind, 
                hist_humidity=50, hist_cloud=50, enso=0, nao=0, pdo=0, variable='rain'):
        """
        Predict using statistical models - SUPPORTS ALL 11 VARIABLES
        
        Args:
            lat, lon: Location
            day_of_year: Day of year (1-366)
            hist_temp, hist_precip, hist_wind: Historical averages
            hist_humidity, hist_cloud: Optional historical averages
            enso, nao, pdo: Climate indices
            variable: One of 11 weather variables
        
        Returns:
            Float probability [0, 1]
        """
        # Build feature vector matching training
        # Features: lat, lon, day_of_year, temp, precip, wind, humidity, cloud, enso, nao, pdo
        features = np.array([[
            lat, lon, day_of_year,
            hist_temp,
            hist_precip,
            hist_wind,
            hist_humidity,
            hist_cloud,
            enso, nao, pdo
        ]])
        
        # Check if variable has trained models
        if variable not in self.models:
            # Fallback to rain if variable not found
            if 'rain' in self.models:
                variable = 'rain'
            else:
                return 0.1  # Default fallback
        
        # Get models for this variable
        model_dict = self.models[variable]
        rf = model_dict['rf']
        xgb = model_dict['xgb']
        
        # Predict
        pred_rf = rf.predict(features)[0]
        pred_xgb = xgb.predict(features)[0]
        
        # Ensemble average
        pred = (pred_rf + pred_xgb) / 2.0
        
        return float(np.clip(pred, 0, 1))
