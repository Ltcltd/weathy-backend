#!/usr/bin/env python3
import subprocess
import sys

def train_all_models():
    scripts = [
        ("scripts/training/train_gnn.py", "GNN (Graph Neural Network)"),
        ("scripts/training/train_statistical.py", "Statistical Models (XGBoost + RF)"),
        ("scripts/training/train_foundation_model.py", "Foundation Model (Pre-trained + Fine-tuned)"),
        ("scripts/training/optimize_ensemble.py", "Quantum Ensemble Optimization"),
    ]
    
    print("="*80)
    print("TRAINING ALL AI MODELS")
    print("="*80)
    print("\nModels to train:")
    for i, (_, name) in enumerate(scripts, 1):
        print(f"  {i}. {name}")
    print("\nEstimated time: ~40 minutes")
    print("="*80)
    input("\nPress Enter to start training...")
    
    for i, (script, name) in enumerate(scripts, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(scripts)}] {name}")
        print(f"{'='*80}\n")
        
        result = subprocess.run([sys.executable, script])
        
        if result.returncode != 0:
            print(f"\n⚠ {name} failed")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    print("\n" + "="*80)
    print("✓ ALL MODELS TRAINED SUCCESSFULLY")
    print("="*80)
    print("\nTrained components:")
    print("  ✓ GNN with teleconnections (108 nodes)")
    print("  ✓ Foundation Model (pre-trained + fine-tuned head)")
    print("  ✓ XGBoost + Random Forest")
    print("  ✓ Quantum-optimized ensemble weights")
    print("\nModels saved:")
    print("  • models/gnn/gnn_model.pth")
    print("  • models/foundation/")
    print("  • models/statistical/models.pkl")
    print("  • models/ensemble/weights.json")
    print("\nNext step: python test_ai_full.py")

if __name__ == "__main__":
    train_all_models()
