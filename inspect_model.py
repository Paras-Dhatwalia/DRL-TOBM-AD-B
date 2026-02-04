"""
Model Inspector - Check what's inside a trained model file

Usage:
    python inspect_model.py path/to/model.pt
    python inspect_model.py models/ppo_billboard_ea.pt
"""

import torch
import sys
from pathlib import Path
from datetime import datetime
import os


def inspect_model(model_path: str):
    """Inspect a saved model checkpoint."""
    path = Path(model_path)

    if not path.exists():
        print(f"ERROR: File not found: {model_path}")
        return

    # File info
    stat = path.stat()
    mod_time = datetime.fromtimestamp(stat.st_mtime)

    print("=" * 60)
    print("MODEL INSPECTION")
    print("=" * 60)
    print(f"File: {path.name}")
    print(f"Path: {path.absolute()}")
    print(f"Size: {stat.st_size / 1024 / 1024:.2f} MB")
    print(f"Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    print("Checkpoint Contents:")
    print("-" * 40)
    for key in checkpoint.keys():
        value = checkpoint[key]
        if isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} with {len(value)} items")
        elif hasattr(value, 'shape'):
            print(f"  {key}: tensor {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")
    print()

    # Model configuration
    if 'config' in checkpoint:
        print("Model Configuration:")
        print("-" * 40)
        cfg = checkpoint['config']
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print()

    # Training configuration
    if 'training_config' in checkpoint:
        print("Training Configuration:")
        print("-" * 40)
        tcfg = checkpoint['training_config']
        for k, v in tcfg.items():
            print(f"  {k}: {v}")
        print()

    # Performance metrics
    print("Performance:")
    print("-" * 40)
    if 'best_reward' in checkpoint:
        print(f"  Best Reward: {checkpoint['best_reward']:.2f}")
    if 'mode' in checkpoint:
        print(f"  Mode: {checkpoint['mode']}")
    print()

    # Model state dict info
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"Model Parameters: {total_params:,}")
        print(f"Number of tensors: {len(state_dict)}")

    print("=" * 60)


def list_models(directory: str = "models"):
    """List all model files in a directory."""
    path = Path(directory)
    if not path.exists():
        print(f"Directory not found: {directory}")
        return

    model_files = list(path.glob("*.pt")) + list(path.glob("*.pth"))

    if not model_files:
        print(f"No model files found in {directory}")
        return

    print(f"Found {len(model_files)} model files in {directory}:")
    print("-" * 60)

    for f in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
        stat = f.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        size_mb = stat.st_size / 1024 / 1024

        # Try to get mode from filename or checkpoint
        mode = "?"
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
            if 'mode' in ckpt:
                mode = ckpt['mode'].upper()
            elif 'config' in ckpt and 'mode' in ckpt['config']:
                mode = ckpt['config']['mode'].upper()
        except:
            pass

        print(f"  {f.name:<35} | {mode:>3} | {size_mb:>5.1f}MB | {mod_time.strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python inspect_model.py <model_path>     # Inspect a specific model")
        print("  python inspect_model.py --list [dir]     # List models in directory")
        print()
        print("Examples:")
        print("  python inspect_model.py models/ppo_billboard_ea.pt")
        print("  python inspect_model.py --list models/")
        sys.exit(1)

    if sys.argv[1] == "--list":
        directory = sys.argv[2] if len(sys.argv) > 2 else "models"
        list_models(directory)
    else:
        inspect_model(sys.argv[1])
