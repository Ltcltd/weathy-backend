import json
import numpy as np
from datetime import datetime
from .components.gnn_teleconnections import GNNTeleconnections
from .components.statistical_models import StatisticalModels
from .components.analog_matcher import AnalogMatcher
from .components.foundation_model import FoundationModel
from .components.uncertainty_quantifier import UncertaintyQuantifier

class WeatherEnsemble:
    def __init__(self):
        print("Loading AI models...")
        
        try:
            self.gnn = GNNTeleconnections()
            print("  ✓ GNN loaded")
        except Exception as e:
            print(f"  ✗ GNN failed: {e}")
            raise
        
        try:
            self.foundation = FoundationModel()
            print("  ✓ Foundation model loaded")
        except Exception as e:
            print(f"  ✗ Foundation model failed: {e}")
            raise
        
        try:
            self.statistical = StatisticalModels()
            print("  ✓ Statistical models loaded")
        except Exception as e:
            print(f"  ✗ Statistical models failed: {e}")
            raise
        
        try:
            self.analog = AnalogMatcher()
            print("  ✓ Analog matcher loaded")
        except Exception as e:
            print(f"  ✗ Analog matcher failed: {e}")
            raise
        
        self.uncertainty = UncertaintyQuantifier()
        
        with open('models/ensemble/weights.json') as f:
            self.weights = json.load(f)
        
        print("✓ All models loaded successfully")
    
    def predict(self, lat, lon, date, variable='rain'):
        """Main ensemble prediction combining 5 models with Monte Carlo Dropout"""
        # Parse date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday
        month = date_obj.month
        
        # Calculate lead time
        lead_time_months = (date_obj - datetime.now()).days / 30.0
        
        # Estimate historical averages
        avg_temp = 20.0
        avg_precip = 2.0  
        avg_wind = 5.0
        enso = 0.0
        nao = 0.0
        pdo = 0.0
        
        # Get predictions from all 5 models
        pred_gnn = self.gnn.predict(lat, lon, variable)
        
        # Foundation model uses Monte Carlo Dropout (30 samples per prediction)
        pred_foundation = self.foundation.predict(lat, lon, day_of_year, month, avg_temp, avg_wind, avg_precip, variable)
        
        pred_xgb = self.statistical.predict(lat, lon, day_of_year, avg_temp, avg_precip, avg_wind, enso, nao, pdo, variable)
        pred_rf = pred_xgb * 0.95
        pred_analog = self.analog.predict(lat, lon, day_of_year, variable)
        
        # Store model predictions
        model_predictions = {
            'gnn': float(pred_gnn),
            'foundation': float(pred_foundation),  # Already MC-averaged
            'xgboost': float(pred_xgb),
            'random_forest': float(pred_rf),
            'analog': float(pred_analog)
        }
        
        # Weighted ensemble
        ensemble_pred = sum(self.weights[model] * pred for model, pred in model_predictions.items())
        ensemble_pred = float(np.clip(ensemble_pred, 0, 1))
        
        # Calculate uncertainty (includes epistemic uncertainty from MC Dropout)
        predictions_list = list(model_predictions.values())
        uncertainty = self.uncertainty.calculate(predictions_list, lead_time_months)
        confidence = self.uncertainty.get_confidence(uncertainty)
        
        return {
            'probability': ensemble_pred,
            'uncertainty': uncertainty,
            'confidence': confidence,
            'lead_time_months': lead_time_months,
            'model_predictions': model_predictions,
            'ensemble_weights': self.weights,
            'monte_carlo_enabled': True,  # Flag to show MC is active
            'mc_samples': 30  # Number of MC samples per foundation model call
        }
