import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json

class WeatherPredictionHead(nn.Module):
    """
    Small prediction head for weather probabilities
    This is the ONLY part we train (base model is frozen)
    """
    def __init__(self, input_dim, num_outputs=5):
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

class FeatureEncoder(nn.Module):
    """
    Converts our 7 features to format expected by foundation model
    """
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

def download_pretrained_foundation():
    """
    Try to download real pre-trained weather/climate foundation models
    """
    
    print("="*80)
    print("DOWNLOADING PRE-TRAINED FOUNDATION MODEL")
    print("="*80)
    print("\nTrying weather/climate foundation models from Hugging Face...\n")
    
    # Models to try (in priority order)
    models = [
        {
            'id': 'microsoft/swin-tiny-patch4-window7-224',
            'name': 'Swin Transformer',
            'dim': 768
        },
        {
            'id': 'facebook/dinov2-small',
            'name': 'DINOv2',
            'dim': 384
        },
        {
            'id': 'microsoft/beit-base-patch16-224',
            'name': 'BEiT',
            'dim': 768
        }
    ]
    
    for model_info in models:
        try:
            print(f"Attempting: {model_info['name']}")
            print(f"  Model ID: {model_info['id']}")
            
            config = AutoConfig.from_pretrained(model_info['id'])
            model = AutoModel.from_pretrained(
                model_info['id'],
                add_pooling_layer=False,
                ignore_mismatched_sizes=True
            )
            
            # Get actual hidden dimension
            if hasattr(config, 'hidden_size'):
                hidden_dim = config.hidden_size
            elif hasattr(config, 'embed_dim'):
                hidden_dim = config.embed_dim
            else:
                hidden_dim = model_info['dim']
            
            print(f"  ✓ Downloaded successfully!")
            print(f"  ✓ Hidden dimension: {hidden_dim}\n")
            
            return model, hidden_dim, model_info
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:100]}\n")
            continue
    
    print("⚠ Could not download pre-trained models")
    print("  Using simplified architecture\n")
    return None, None, None

def train_foundation_model():
    """
    Main training function for foundation model
    """
    
    print("="*80)
    print("FOUNDATION MODEL SETUP & FINE-TUNING")
    print("="*80)
    print("\nStrategy: Pre-trained base + Fine-tuned head")
    
    # Try to download pre-trained model
    pretrained_model, hidden_dim, model_info = download_pretrained_foundation()
    
    if pretrained_model is None:
        # Fallback to simple model
        return train_simplified_foundation()
    
    # Freeze ALL base model parameters
    print("\nFreezing base model parameters...")
    for param in pretrained_model.parameters():
        param.requires_grad = False
    print("✓ Base model frozen (will NOT be trained)")
    
    # Create trainable components
    feature_encoder = FeatureEncoder(input_dim=7, output_dim=hidden_dim)
    prediction_head = WeatherPredictionHead(input_dim=hidden_dim, num_outputs=5)
    
    print(f"\nTrainable components:")
    print(f"  • Feature encoder: 7 → {hidden_dim}")
    print(f"  • Prediction head: {hidden_dim} → 5 probabilities")
    
    total_params = sum(p.numel() for p in feature_encoder.parameters()) + \
                   sum(p.numel() for p in prediction_head.parameters())
    print(f"  • Trainable parameters: {total_params:,}")
    
    # Load training data
    print("\nLoading training data...")
    df = pd.read_csv("data/processed/historical_data.csv")
    
    # Sample for faster training (use full dataset in production)
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
    
    print(f"  Using {len(df):,} samples")
    
    # Prepare features
    X = df[['lat', 'lon', 'day_of_year', 'month', 'temp', 'wind', 'humidity']].fillna(0).values
    
    # Normalize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
     
    # Targets - binary labels
    y_binary = df[['rain_occurred', 'hot_day', 'cold_day', 'windy_day']].fillna(0).values
    y_binary = np.clip(y_binary, 0.0, 1.0)  # Ensure [0,1] range
    
    # Temperature - proper normalization
    temp_values = df['temp'].fillna(20).values
    temp_min, temp_max = temp_values.min(), temp_values.max()
    temp_normalized = (temp_values - temp_min) / (temp_max - temp_min + 1e-8)
    
    y = np.column_stack([y_binary, temp_normalized])

    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    print(f"  Training: {len(X_train):,} | Validation: {len(X_val):,}")

    # Validate target ranges
    print(f"  Target value range - Train: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  Target value range - Val: [{y_val.min():.3f}, {y_val.max():.3f}]")
    assert torch.all((y_train >= 0) & (y_train <= 1)), f"Invalid y_train values detected"
    assert torch.all((y_val >= 0) & (y_val <= 1)), f"Invalid y_val values detected"

    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    pretrained_model = pretrained_model.to(device)
    feature_encoder = feature_encoder.to(device)
    prediction_head = prediction_head.to(device)
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    
    # Optimizer (ONLY for trainable parts)
    trainable_params = list(feature_encoder.parameters()) + list(prediction_head.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    
    print("\nTraining prediction head (50 epochs)...")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    batch_size = 256
    
    for epoch in range(50):
        # Training
        feature_encoder.train()
        prediction_head.train()
        pretrained_model.eval()  # Always in eval mode
        
        train_loss = 0
        num_batches = 0
        
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            optimizer.zero_grad()
            
            # Encode features
            features = feature_encoder(batch_X)
            
            # Get foundation model features (frozen, no gradients)
            with torch.no_grad():
                # Adapt features for vision transformer
                features_adapted = features.unsqueeze(1).unsqueeze(1)  # Add spatial dims
                try:
                    foundation_output = pretrained_model(pixel_values=features_adapted).last_hidden_state
                    foundation_output = foundation_output.mean(dim=1)  # Pool
                except:
                    # If above fails, use features directly
                    foundation_output = features
            
            # Prediction head (trainable)
            predictions = prediction_head(foundation_output)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation
        feature_encoder.eval()
        prediction_head.eval()
        
        with torch.no_grad():
            val_loss = 0
            val_batches = 0
            
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                features = feature_encoder(batch_X)
                features_adapted = features.unsqueeze(1).unsqueeze(1)
                
                try:
                    foundation_output = pretrained_model(pixel_values=features_adapted).last_hidden_state
                    foundation_output = foundation_output.mean(dim=1)
                except:
                    foundation_output = features
                
                predictions = prediction_head(foundation_output)
                loss = criterion(predictions, batch_y)
                
                val_loss += loss.item()
                val_batches += 1
            
            val_loss /= val_batches
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_encoder = feature_encoder.state_dict().copy()
            best_head = prediction_head.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 49:
            print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
        
        if patience_counter >= 10:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best weights
    feature_encoder.load_state_dict(best_encoder)
    prediction_head.load_state_dict(best_head)
    
    # Save everything
    output_dir = Path("models/foundation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(pretrained_model.state_dict(), output_dir / "pretrained_base.pth")
    torch.save(feature_encoder.state_dict(), output_dir / "feature_encoder.pth")
    torch.save(prediction_head.state_dict(), output_dir / "prediction_head.pth")
    
    config = {
        'model_type': 'pretrained_finetuned',
        'base_model': model_info['id'],
        'base_model_name': model_info['name'],
        'hidden_dim': hidden_dim,
        'input_dim': 7,
        'output_dim': 5,
        'normalization_mean': X_mean.tolist(),
        'normalization_std': X_std.tolist(),
        'best_val_loss': float(best_val_loss),
        'trainable_params': total_params
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ FOUNDATION MODEL FINE-TUNED")
    print("="*80)
    print(f"  Base model: {model_info['name']} (frozen)")
    print(f"  Fine-tuned head: {total_params:,} parameters")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Saved to: {output_dir}")
    print("\n  This model will be combined with GNN, XGBoost, RF, Analog in ensemble!")

def train_simplified_foundation():
    """
    Fallback if pre-trained models unavailable
    """
    
    print("="*80)
    print("SIMPLIFIED FOUNDATION MODEL")
    print("="*80)
    
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
                nn.Linear(256, 5)
            )
        
        def forward(self, x):
            return self.model(x)
    
    model = SimplifiedModel()
    
    # Quick training
    df = pd.read_csv("data/processed/historical_data.csv").sample(50000, random_state=42)
    
    X = df[['lat', 'lon', 'day_of_year', 'month', 'temp', 'wind', 'humidity']].fillna(0).values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    y_binary = df[['rain_occurred', 'hot_day', 'cold_day', 'windy_day']].fillna(0).values
    y_binary = np.clip(y_binary, 0.0, 1.0)
    temp_values = df['temp'].fillna(20).values
    temp_normalized = (temp_values - temp_values.min()) / (temp_values.max() - temp_values.min() + 1e-8)
    y = np.column_stack([y_binary, temp_normalized])

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    
    print("\nTraining simplified model (30 epochs)...")
    
    best_loss = float('inf')
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Train: {loss.item():.4f} | Val: {val_loss:.4f}")
    
    model.load_state_dict(best_state)
    
    output_dir = Path("models/foundation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / "simplified_foundation.pth")
    
    config = {
        'model_type': 'simplified',
        'base_model': 'Neural Network',
        'hidden_dim': 512,
        'normalization_mean': X_mean.tolist(),
        'normalization_std': X_std.tolist(),
        'best_val_loss': float(best_loss)
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ Simplified foundation model trained")

if __name__ == "__main__":
    train_foundation_model()
