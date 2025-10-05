import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path

class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=7, output_dim=768):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class WeatherPredictionHead(nn.Module):
    def __init__(self, input_dim, num_outputs=11):  # CHANGED from 5 to 11
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_outputs)
        )
    
    def forward(self, x):
        return self.head(x)

class SimplifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(7, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 11)  # CHANGED from 5 to 11
        )
    
    def forward(self, x):
        return torch.sigmoid(self.model(x))

class FoundationModel:
    """Foundation Model for Ensemble - Pre-trained + Fine-tuned"""
    
    # Map API variables to output indices (11 total)
    VARIABLE_INDEX = {
        'rain': 0,
        'heavy_rain': 1,
        'snow': 2,
        'cloud_cover': 3,
        'wind_speed_high': 4,
        'temperature_hot': 5,
        'temperature_cold': 6,
        'heat_wave': 7,
        'cold_snap': 8,
        'dust_event': 9,
        'uncomfortable_index': 10
    }
    
    def __init__(self):
        config_path = Path('models/foundation/config.json')
        
        if not config_path.exists():
            # Fallback to simplified model if config missing
            config_path = Path('models/foundation/normalization.json')
        
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.mean = np.array(self.config['normalization_mean'])
        self.std = np.array(self.config['normalization_std'])
        
        if self.config.get('model_type') == 'simplified':
            # Simplified model
            self.model = SimplifiedModel()
            model_path = Path('models/foundation/foundation_model.pth')
            if not model_path.exists():
                model_path = Path('models/foundation/simplified_foundation.pth')
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.model.eval()
            self.mode = 'simplified'
        else:
            # Pre-trained + fine-tuned
            self.feature_encoder = FeatureEncoder(
                input_dim=7,
                output_dim=self.config['hidden_dim']
            )
            self.feature_encoder.load_state_dict(torch.load('models/foundation/feature_encoder.pth', weights_only=True))
            self.feature_encoder.eval()
            
            self.prediction_head = WeatherPredictionHead(
                input_dim=self.config['hidden_dim'],
                num_outputs=11  # CHANGED from 5 to 11
            )
            self.prediction_head.load_state_dict(torch.load('models/foundation/prediction_head.pth', weights_only=True))
            self.prediction_head.eval()
            self.mode = 'pretrained'
        
        print(f"Foundation Model loaded: {self.config.get('base_model_name', 'Simplified')}")
    
    def predict(self, lat, lon, day_of_year, month, temp, wind, humidity, variable='rain'):
        """
        Get prediction from foundation model using Monte Carlo Dropout
        Uses 30 stochastic forward passes for epistemic uncertainty quantification
        Returns mean prediction from MC sampling
        """
        # Prepare features
        features = np.array([[lat, lon, day_of_year, month, temp, wind, humidity]])
        features = (features - self.mean) / self.std
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Get variable index
        var_idx = self.VARIABLE_INDEX.get(variable, 0)
        
        # Monte Carlo Dropout - enable dropout during inference
        predictions = []
        n_samples = 30
        
        if self.mode == 'simplified':
            self.model.train()  # Enable dropout
        else:
            self.feature_encoder.train()
            self.prediction_head.train()
        
        for _ in range(n_samples):
            with torch.no_grad():
                if self.mode == 'simplified':
                    pred = self.model(features_tensor)
                else:
                    encoded = self.feature_encoder(features_tensor)
                    logits = self.prediction_head(encoded)
                    pred = torch.sigmoid(logits)
                
                predictions.append(float(pred[0, var_idx].item()))
        
        # Back to eval mode
        if self.mode == 'simplified':
            self.model.eval()
        else:
            self.feature_encoder.eval()
            self.prediction_head.eval()
        
        # Return mean of MC samples
        return float(np.mean(predictions))
    
    def predict_with_uncertainty(self, lat, lon, day_of_year, month, temp, wind, humidity, variable='rain'):
        """
        Monte Carlo Dropout with full uncertainty quantification
        Returns:
            dict with mean, epistemic_uncertainty, and confidence_interval_95
        """
        # Prepare features
        features = np.array([[lat, lon, day_of_year, month, temp, wind, humidity]])
        features = (features - self.mean) / self.std
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Get variable index
        var_idx = self.VARIABLE_INDEX.get(variable, 0)
        
        # Monte Carlo sampling
        predictions = []
        n_samples = 30
        
        if self.mode == 'simplified':
            self.model.train()
        else:
            self.feature_encoder.train()
            self.prediction_head.train()
        
        for _ in range(n_samples):
            with torch.no_grad():
                if self.mode == 'simplified':
                    pred = self.model(features_tensor)
                else:
                    encoded = self.feature_encoder(features_tensor)
                    logits = self.prediction_head(encoded)
                    pred = torch.sigmoid(logits)
                
                predictions.append(float(pred[0, var_idx].item()))
        
        # Back to eval mode
        if self.mode == 'simplified':
            self.model.eval()
        else:
            self.feature_encoder.eval()
            self.prediction_head.eval()
        
        return {
            'mean': float(np.mean(predictions)),
            'epistemic_uncertainty': float(np.std(predictions)),
            'confidence_interval_95': (
                float(np.percentile(predictions, 2.5)),
                float(np.percentile(predictions, 97.5))
            ),
            'n_samples': n_samples
        }
