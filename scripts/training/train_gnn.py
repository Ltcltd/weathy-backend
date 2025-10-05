import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

class GNNModel(torch.nn.Module):
    def __init__(self, num_features=10, hidden=128, output=11):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, output)
        self.dropout = torch.nn.Dropout(0.5)
        self.batch_norm1 = torch.nn.BatchNorm1d(hidden)
        self.batch_norm2 = torch.nn.BatchNorm1d(hidden)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

def train_gnn():
    print("=" * 80)
    print("TRAINING GRAPH NEURAL NETWORK - 11 OUTPUT VARIABLES")
    print("=" * 80)
    
    # Load graph
    with open("data/processed/graphs/climate_graph.pkl", 'rb') as f:
        graph_data = pickle.load(f)
    
    graph = graph_data['graph']
    regions = graph_data['regions']
    
    print(f"\nGraph structure:")
    print(f"  Nodes: {len(regions)} climate regions")
    print(f"  Edges: {graph.edge_index.size(1)} teleconnections")
    
    # Load historical data
    df = pd.read_csv("data/processed/historical_probabilities.csv")
    print(f"\nHistorical data: {len(df):,} records")
    
    # Load climate indices
    try:
        enso = pd.read_csv("data/raw/climate_indices/enso.csv")
        enso_avg = enso[enso['Year'] == 2023]['Value'].mean()
    except:
        enso_avg = 0
    
    print("\nPreparing node features and targets...")
    
    # Define target column names
    target_columns = [
        'rain_mean',
        'heavy_rain_mean',
        'snow_mean',
        'cloud_cover_high_mean',
        'wind_speed_high_mean',
        'temperature_hot_mean',
        'temperature_cold_mean',
        'heat_wave_mean',
        'cold_snap_mean',
        'dust_event_mean',
        'uncomfortable_index_mean'
    ]
    
    # Check which targets exist
    available_targets = [col for col in target_columns if col in df.columns]
    missing_targets = [col for col in target_columns if col not in df.columns]
    
    if missing_targets:
        print(f"\nWarning: Missing target columns: {missing_targets}")
    print(f"Using {len(available_targets)} target variables")
    
    # Fill graph with actual data
    for idx, region in enumerate(regions):
        region_data = df[(df['lat'] == region['lat']) & (df['lon'] == region['lon'])]
        
        if len(region_data) > 0:
            # Features: historical averages and climate info
            avg_temp = region_data['temp_mean'].mean() if 'temp_mean' in df.columns else 20.0
            avg_precip = region_data['precip_mean'].mean() if 'precip_mean' in df.columns else 2.0
            avg_wind = region_data['wind_mean'].mean() if 'wind_mean' in df.columns else 5.0
            
            # Normalize features to [0, 1] range
            graph.x[idx, 0] = np.clip(avg_temp / 50.0, 0, 1)
            graph.x[idx, 1] = np.clip(avg_precip / 10.0, 0, 1)
            graph.x[idx, 2] = np.clip(avg_wind / 20.0, 0, 1)
            
            # Climate index
            graph.x[idx, 3] = (enso_avg + 3) / 6  # Normalize ENSO to [0,1]
            
            # Spatial features
            graph.x[idx, 6] = (region['lat'] + 90) / 180  # Normalize to [0,1]
            graph.x[idx, 7] = (region['lon'] + 180) / 360
            
            # Set all 11 targets (probability of each event type)
            for target_idx, col_name in enumerate(target_columns):
                if col_name in df.columns:
                    value = region_data[col_name].mean()
                    graph.y[idx, target_idx] = np.clip(value, 0, 1)
                else:
                    # Fallback for missing columns
                    graph.y[idx, target_idx] = 0.1
    
    # Initialize model with 11 outputs
    model = GNNModel(num_features=10, hidden=128, output=11)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    print("\nTraining GNN for 200 epochs with regularization...")
    print("-" * 80)
    
    model.train()
    best_loss = float('inf')
    patience = 0
    max_patience = 20
    
    for epoch in range(200):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(graph.x, graph.edge_index)
        
        # Calculate loss
        loss = F.mse_loss(out, graph.y)
        
        # L1 regularization on outputs (encourage sparsity)
        l1_reg = 0.01 * torch.mean(torch.abs(out))
        total_loss = loss + l1_reg
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        # Print progress
        if epoch % 20 == 0 or epoch == 199:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Best: {best_loss:.4f} | Patience: {patience}/{max_patience}")
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Save model
    output_dir = Path("models/gnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / "gnn_model.pth")
    torch.save({
        'num_features': 10,
        'hidden': 128,
        'output': 11,
        'final_loss': loss.item(),
        'best_loss': best_loss,
        'target_columns': target_columns
    }, output_dir / "gnn_config.pth")
    
    print("\n" + "=" * 80)
    print("GNN MODEL TRAINED - 11 OUTPUT VARIABLES")
    print("=" * 80)
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Output variables: {len(target_columns)}")
    print(f"  Model saved: models/gnn/gnn_model.pth")
    print(f"\n  Target variables:")
    for i, col in enumerate(target_columns, 1):
        print(f"    {i:2d}. {col}")
    print("=" * 80)

if __name__ == "__main__":
    train_gnn()
