import torch
from torch_geometric.data import Data
import pickle
from pathlib import Path
import numpy as np

def build_teleconnection_graph():
    """Build climate teleconnection graph for GNN"""
    
    # Define climate regions (18 lats x 6 lons = 108 regions)
    regions = []
    region_map = {}
    idx = 0
    
    for lat in range(-90, 100, 10):  # Every 10 degrees
        for lon in range(-180, 181, 60):  # Every 60 degrees
            regions.append({'lat': lat, 'lon': lon, 'idx': idx})
            region_map[(lat, lon)] = idx
            idx += 1
    
    num_regions = len(regions)
    print(f"Creating graph with {num_regions} climate regions")
    
    # Build edges (connections between regions)
    edges = []
    
    for i, reg1 in enumerate(regions):
        for j, reg2 in enumerate(regions):
            if i == j:
                continue
            
            lat_diff = abs(reg1['lat'] - reg2['lat'])
            lon_diff = abs(reg1['lon'] - reg2['lon'])
            
            # 1. Connect adjacent regions
            if lat_diff <= 10 and lon_diff <= 60:
                edges.append([i, j])
            
            # 2. ENSO teleconnection: Tropical Pacific → Americas
            if (-10 <= reg1['lat'] <= 10 and 120 <= reg1['lon'] <= 180):
                if (30 <= reg2['lat'] <= 50 and -120 <= reg2['lon'] <= -60):
                    edges.append([i, j])
            
            # 3. NAO teleconnection: North Atlantic → Europe
            if (40 <= reg1['lat'] <= 60 and -60 <= reg1['lon'] <= 0):
                if (40 <= reg2['lat'] <= 60 and 0 <= reg2['lon'] <= 60):
                    edges.append([i, j])
            
            # 4. IOD teleconnection: Indian Ocean → Africa/Asia
            if (-10 <= reg1['lat'] <= 10 and 60 <= reg1['lon'] <= 120):
                if (0 <= reg2['lat'] <= 30 and 0 <= reg2['lon'] <= 60):
                    edges.append([i, j])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Node features (10 features per region)
    # [avg_temp, avg_precip, avg_wind, enso, nao, pdo, lat_norm, lon_norm, season_sin, season_cos]
    x = torch.randn(num_regions, 10) * 0.1
    
    # Targets (11 weather event types) - FIXED from 5 to 11
    y = torch.zeros(num_regions, 11)
    
    graph = Data(x=x, edge_index=edge_index, y=y)
    
    # Save
    output_dir = Path("data/processed/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graph_data = {
        'graph': graph,
        'regions': regions,
        'region_map': region_map
    }
    
    with open(output_dir / "climate_graph.pkl", 'wb') as f:
        pickle.dump(graph_data, f)
    
    print(f"Graph: {num_regions} nodes, {len(edges)} edges")
    print(f"Teleconnections: ENSO, NAO, IOD included")
    
    return graph

if __name__ == "__main__":
    build_teleconnection_graph()
