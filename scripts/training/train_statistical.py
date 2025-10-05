import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
from pathlib import Path

def train_statistical_models():
    """
    Train statistical models (Random Forest + XGBoost) for ALL 11 weather variables
    """
    print("=" * 80)
    print("TRAINING STATISTICAL MODELS - ALL 11 VARIABLES")
    print("=" * 80)
    
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
        
        # Add climate indices (use 2023 as reference year)
        enso_2023 = enso[enso['Year'] == 2023][['Month', 'Value']].rename(columns={'Value': 'enso'})
        nao_2023 = nao[nao['Year'] == 2023][['Month', 'Value']].rename(columns={'Value': 'nao'})
        pdo_2023 = pdo[pdo['Year'] == 2023][['Month', 'Value']].rename(columns={'Value': 'pdo'})
        
        df = df.merge(enso_2023, left_on='month', right_on='Month', how='left')
        df = df.merge(nao_2023, left_on='month', right_on='Month', how='left', suffixes=('', '_nao'))
        df = df.merge(pdo_2023, left_on='month', right_on='Month', how='left', suffixes=('', '_pdo'))
        
        df['enso'] = df['enso'].fillna(0)
        df['nao'] = df['nao'].fillna(0)
        df['pdo'] = df['pdo'].fillna(0)
        
        print("   ✓ Climate indices added (ENSO, NAO, PDO)")
        
    except Exception as e:
        # Fallback if climate indices fail
        df['enso'] = 0
        df['nao'] = 0
        df['pdo'] = 0
        print(f"Using default climate indices (failed: {e})")
    
    # Feature columns: location, time, historical averages, climate indices
    feature_cols = [
        'lat', 'lon', 'day_of_year',
        'temp_mean',    # Historical temperature average
        'precip_mean',  # Historical precipitation average
        'wind_mean',    # Historical wind average
        'humidity_mean', # Historical humidity average
        'cloud_mean',   # Historical cloud cover average
        'enso', 'nao', 'pdo'  # Climate indices
    ]
    
    # Check which feature columns exist
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"Missing features: {missing_features}")
        print(f"   Using {len(available_features)} features: {available_features}")
    
    X = df[available_features].fillna(0).values
    
    # ALL 11 TARGETS - map to column names from prepare_training_data.py
    targets = {
        'rain': 'rain_mean',
        'heavy_rain': 'heavy_rain_mean',
        'snow': 'snow_mean',
        'cloud_cover_high': 'cloud_cover_high_mean',
        'wind_speed_high': 'wind_speed_high_mean',
        'temperature_hot': 'temperature_hot_mean',
        'temperature_cold': 'temperature_cold_mean',
        'heat_wave': 'heat_wave_mean',
        'cold_snap': 'cold_snap_mean',
        'dust_event': 'dust_event_mean',
        'uncomfortable_index': 'uncomfortable_index_mean'
    }
    
    print(f"\nTraining targets: {len(targets)} weather variables")
    for i, (key, col) in enumerate(targets.items(), 1):
        if col in df.columns:
            print(f"   {i:2d}. {key:25s} → {col}")
        else:
            print(f"   {i:2d}. {key:25s} → MISSING")
    
    models = {}
    
    for target_name, target_col in targets.items():
        # Check if target exists
        if target_col not in df.columns:
            print(f"\nSkipping {target_name} - column '{target_col}' not found")
            continue
        
        y = df[target_col].values
        
        print(f"\n{'=' * 80}")
        print(f"TRAINING: {target_name.upper()}")
        print(f"{'=' * 80}")
        print(f"   Target: {target_col}")
        print(f"   Mean probability: {y.mean():.4f}")
        print(f"   Std: {y.std():.4f}")
        
        # Proper train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Random Forest with regularization
        print("\n   [1/2] Training Random Forest...")
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
        
        print(f"      Train R²: {train_score:.4f}")
        print(f"      Test R²:  {test_score:.4f}")
        
        # XGBoost with regularization
        print("\n   [2/2] Training XGBoost...")
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
        
        print(f"      Train R²: {train_score_xgb:.4f}")
        print(f"      Test R²:  {test_score_xgb:.4f}")
        
        # Store models and scores
        models[target_name] = {
            'rf': rf,
            'xgb': xgb_model,
            'rf_train_score': train_score,
            'rf_test_score': test_score,
            'xgb_train_score': train_score_xgb,
            'xgb_test_score': test_score_xgb,
            'target_col': target_col,
            'mean_prob': y.mean()
        }
    
    # Save models
    output_dir = Path("models/statistical")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "models.pkl", 'wb') as f:
        pickle.dump(models, f)
    
    # Save feature names for inference
    with open(output_dir / "feature_names.pkl", 'wb') as f:
        pickle.dump(available_features, f)
    
    print("\n" + "=" * 80)
    print("STATISTICAL MODELS TRAINED")
    print("=" * 80)
    
    print(f"\nModel Performance Summary ({len(models)} variables):\n")
    print(f"{'Variable':<25} {'RF Test R²':<12} {'XGB Test R²':<12} {'Mean Prob':<12}")
    print("-" * 80)
    
    for target, model_info in sorted(models.items()):
        rf_test = model_info['rf_test_score']
        xgb_test = model_info['xgb_test_score']
        mean_prob = model_info['mean_prob']
        print(f"{target:<25} {rf_test:<12.4f} {xgb_test:<12.4f} {mean_prob:<12.4f}")
    
    print("\n" + "=" * 80)
    print(f"Models saved to: models/statistical/")
    print(f"   • models.pkl: {len(models)} target variables × 2 models (RF + XGB)")
    print(f"   • feature_names.pkl: {len(available_features)} feature names")
    print("=" * 80)


if __name__ == "__main__":
    train_statistical_models()
