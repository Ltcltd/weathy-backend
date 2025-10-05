#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_preprocessing():
    scripts = [
        ("/code/scripts/preprocessing/prepare_training_data.py", "Preparing training data"),
        ("/code/scripts/preprocessing/build_teleconnection_graph.py", "Building GNN graph"),
        ("/code/scripts/preprocessing/build_global_grids.py", "Building probability grids"),
        ("/code/scripts/preprocessing/integrate_sources.py", "Integrating data sources"),
    ]
    
    print("="*80)
    print("PREPROCESSING PIPELINE")
    print("="*80)
    print(f"\nTotal steps: {len(scripts)}\n")
    
    for i, (script, desc) in enumerate(scripts, 1):
        print(f"[{i}/{len(scripts)}] {desc}...")
        print("-"*80)
        
        result = subprocess.run([sys.executable, script])
        
        if result.returncode != 0:
            print(f"\n⚠ Warning: {desc} had issues")
            response = input("Continue? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        print()
    
    print("="*80)
    print("✓ PREPROCESSING COMPLETE")
    print("="*80)
    
    # Check outputs
    print("\nGenerated files:")
    if Path("data/processed/historical_data.csv").exists():
        print("  ✓ historical_data.csv")
    if Path("data/processed/historical_probabilities.csv").exists():
        print("  ✓ historical_probabilities.csv")
    if Path("data/processed/graphs/climate_graph.pkl").exists():
        print("  ✓ climate_graph.pkl")
    if Path("data/processed/global_grids").exists():
        grids = list(Path("data/processed/global_grids").glob("*.h5"))
        print(f"  ✓ {len(grids)} probability grids")
    if Path("data/processed/data_integration_summary.json").exists():
        print("  ✓ data_integration_summary.json")
    
    print("\nNext step: python train_all.py")

if __name__ == "__main__":
    run_preprocessing()
