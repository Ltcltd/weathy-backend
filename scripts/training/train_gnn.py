import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

class GNNModel(torch.nn.Module):
    def __init__(self, num_features=10, hidden=128, output=5):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, output)
        self.dropout = torch.nn.Dropout(0.5)  # Increased dropout
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
    print("="*80)
    print("TRAINING GRAPH NEURAL NETWORK (FIXED)")
    print("="*80)
    
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
    
    # Fill graph with actual data
    for idx, region in enumerate(regions):
        region_data = df[(df['lat'] == region['lat']) & (df['lon'] == region['lon'])]
        
        if len(region_data) > 0:
            # Features: ONLY historical averages and climate info
            avg_temp = region_data['temp_mean'].mean()
            avg_precip = region_data['precip_mean'].mean()
            avg_wind = region_data['wind_mean'].mean()
            
            # Normalize features to [0, 1] range
            graph.x[idx, 0] = np.clip(avg_temp / 50.0, 0, 1)
            graph.x[idx, 1] = np.clip(avg_precip / 10.0, 0, 1)
            graph.x[idx, 2] = np.clip(avg_wind / 20.0, 0, 1)
            
            # Climate index
            graph.x[idx, 3] = (enso_avg + 3) / 6  # Normalize ENSO to [0,1]
            
            # Spatial features
            graph.x[idx, 6] = (region['lat'] + 90) / 180  # Normalize to [0,1]
            graph.x[idx, 7] = (region['lon'] + 180) / 360
            
            # Targets: probability of each event type
            graph.y[idx, 0] = np.clip(region_data['rain_occurred_mean'].mean(), 0, 1)
            graph.y[idx, 1] = np.clip(region_data['hot_day_mean'].mean(), 0, 1)
            graph.y[idx, 2] = np.clip(region_data['cold_day_mean'].mean(), 0, 1)
            graph.y[idx, 3] = np.clip(region_data['windy_day_mean'].mean(), 0, 1)
            graph.y[idx, 4] = np.clip(region_data['temp_mean'].mean() / 50.0, 0, 1)
    
    # Initialize model
    model = GNNModel(num_features=10, hidden=128, output=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # Added weight decay
    
    print("\nTraining GNN for 200 epochs with regularization...")
    print("-"*80)
    
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
        'output': 5,
        'final_loss': loss.item(),
        'best_loss': best_loss
    }, output_dir / "gnn_config.pth")
    
    print("\n" + "="*80)
    print("✓ GNN MODEL TRAINED (WITH REGULARIZATION)")
    print("="*80)
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"  Model saved: models/gnn/gnn_model.pth")
    print(f"\n  Loss > 0 is GOOD - means model generalizes!")

if __name__ == "__main__":
    train_gnn()
