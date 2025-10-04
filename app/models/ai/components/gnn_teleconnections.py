import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pickle
import numpy as np

class GNNModel(torch.nn.Module):
    def __init__(self, num_features=10, hidden=128, output=5):
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
    def __init__(self):
        self.model = GNNModel()
        self.model.load_state_dict(torch.load('models/gnn/gnn_model.pth', weights_only=True))
        self.model.eval()
        
        with open('data/processed/graphs/climate_graph.pkl', 'rb') as f:
            data = pickle.load(f)
            self.graph = data['graph']
            self.regions = data['regions']
            self.region_map = data['region_map']
    
    def predict(self, lat, lon, variable='rain'):
        """Predict using GNN"""
        region_idx = self._find_nearest_region(lat, lon)
        with torch.no_grad():
            out = self.model(self.graph.x, self.graph.edge_index)
        
        var_idx = {'rain': 0, 'hot': 1, 'cold': 2, 'windy': 3, 'temp': 4}.get(variable, 0)
        return float(out[region_idx, var_idx].item())
    
    def _find_nearest_region(self, lat, lon):
        """Find nearest region node"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for region in self.regions:
            dist = ((region['lat'] - lat)**2 + (region['lon'] - lon)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_idx = region['idx']
        
        return nearest_idx
