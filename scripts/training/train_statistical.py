import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from pathlib import Path

def train_statistical_models():
    print("="*80)
    print("TRAINING STATISTICAL MODELS")
    print("="*80)
    
    # Load data
    df = pd.read_csv("data/processed/historical_probabilities.csv")
    print(f"\nTraining on {len(df):,} location-day combinations")
    
    # Load climate indices
    try:
        enso = pd.read_csv("data/raw/climate_indices/enso.csv")
        nao = pd.read_csv("data/raw/climate_indices/nao.csv")
        pdo = pd.read_csv("data/raw/climate_indices/pdo.csv")
        
        # Create month index for merging
        df['month'] = ((df['day_of_year'] - 1) // 30) + 1
        df['month'] = df['month'].clip(1, 12)
        
        # Add climate indices (use 2023 as reference)
        enso_2023 = enso[enso['Year'] == 2023][['Month', 'Value']].rename(columns={'Value': 'enso'})
        nao_2023 = nao[nao['Year'] == 2023][['Month', 'Value']].rename(columns={'Value': 'nao'})
        pdo_2023 = pdo[pdo['Year'] == 2023][['Month', 'Value']].rename(columns={'Value': 'pdo'})
        
        df = df.merge(enso_2023, left_on='month', right_on='Month', how='left')
        df = df.merge(nao_2023, left_on='month', right_on='Month', how='left', suffixes=('', '_nao'))
        df = df.merge(pdo_2023, left_on='month', right_on='Month', how='left', suffixes=('', '_pdo'))
        
        df['enso'] = df['enso'].fillna(0)
        df['nao'] = df['nao'].fillna(0)
        df['pdo'] = df['pdo'].fillna(0)
        
        print("  ✓ Climate indices added")
    except:
        # Fallback if climate indices fail
        df['enso'] = 0
        df['nao'] = 0
        df['pdo'] = 0
        print("  ⚠ Using default climate indices")
    
    # FIXED FEATURES: Only location, time, historical averages, climate indices
    feature_cols = [
        'lat', 'lon', 'day_of_year', 
        'temp_mean',      # Historical average, not current!
        'precip_mean',    # Historical average
        'wind_mean',      # Historical average
        'enso', 'nao', 'pdo'  # Climate indices
    ]
    
    X = df[feature_cols].fillna(0).values
    
    # Targets (use the _mean columns from historical probabilities)
    targets = {
        'rain': df['rain_occurred_mean'].values,
        'hot': df['hot_day_mean'].values,
        'cold': df['cold_day_mean'].values,
        'windy': df['windy_day_mean'].values
    }
    
    models = {}
    
    for target_name, y in targets.items():
        print(f"\n{'='*80}")
        print(f"Training models for: {target_name.upper()}")
        print(f"{'='*80}")
        
        # Proper train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Random Forest with regularization
        print("\n[1/2] Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,           # Limit depth to prevent overfitting
            min_samples_split=20,   # Require more samples to split
            min_samples_leaf=10,    # Require more samples per leaf
            max_features='sqrt',    # Use subset of features
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf.fit(X_train, y_train)
        
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²:  {test_score:.4f}")
        
        # XGBoost with regularization
        print("\n[2/2] Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,              # Limit depth
            learning_rate=0.05,       # Lower learning rate
            min_child_weight=5,       # Regularization
            subsample=0.8,            # Use 80% of data per tree
            colsample_bytree=0.8,     # Use 80% of features per tree
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=1.0,           # L2 regularization
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, verbose=False)
        
        train_score_xgb = xgb_model.score(X_train, y_train)
        test_score_xgb = xgb_model.score(X_test, y_test)
        print(f"  Train R²: {train_score_xgb:.4f}")
        print(f"  Test R²:  {test_score_xgb:.4f}")
        
        models[target_name] = {
            'rf': rf,
            'xgb': xgb_model,
            'rf_train_score': train_score,
            'rf_test_score': test_score,
            'xgb_train_score': train_score_xgb,
            'xgb_test_score': test_score_xgb
        }
    
    # Save models
    output_dir = Path("models/statistical")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "models.pkl", 'wb') as f:
        pickle.dump(models, f)
    
    # Save feature names for inference
    with open(output_dir / "feature_names.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("\n" + "="*80)
    print("✓ STATISTICAL MODELS TRAINED")
    print("="*80)
    print("\nModel Performance:")
    for target, model_info in models.items():
        print(f"\n  {target.upper()}:")
        print(f"    Random Forest: Train={model_info['rf_train_score']:.4f}, Test={model_info['rf_test_score']:.4f}")
        print(f"    XGBoost:       Train={model_info['xgb_train_score']:.4f}, Test={model_info['xgb_test_score']:.4f}")
    
    print(f"\n  Models saved: models/statistical/models.pkl")

if __name__ == "__main__":
    train_statistical_models()
