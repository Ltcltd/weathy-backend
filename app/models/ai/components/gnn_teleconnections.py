import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
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

class GNNTeleconnections:
    # Map API variable names to output indices
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
        self.model = GNNModel(output=11)  # 11 outputs
        self.model.load_state_dict(torch.load('models/gnn/gnn_model.pth', weights_only=True, map_location='cpu'))
        self.model.eval()
        
        with open('data/processed/graphs/climate_graph.pkl', 'rb') as f:
            self.graph_data = pickle.load(f)
    
    def predict(self, lat, lon, variable='rain'):
        """
        Predict weather probability using teleconnections
        
        Args:
            lat: Latitude
            lon: Longitude
            variable: One of 11 weather variables
        
        Returns:
            Float probability [0, 1]
        """
        # Find nearest node in graph
        regions = self.graph_data['regions']
        distances = [(i, (r['lat'] - lat)**2 + (r['lon'] - lon)**2) for i, r in enumerate(regions)]
        nearest_idx = min(distances, key=lambda x: x[1])[0]
        
        # Run inference
        graph = self.graph_data['graph']
        with torch.no_grad():
            predictions = self.model(graph.x, graph.edge_index)
        
        # Get correct output index for requested variable
        var_idx = self.VARIABLE_INDEX.get(variable, 0)
        
        return float(predictions[nearest_idx, var_idx].item())
