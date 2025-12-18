import json
import torch
import os
from datetime import datetime

print("=" * 60)
print("TRAINING STATUS CHECK")
print("=" * 60)

# Check checkpoint
if os.path.exists("checkpoints/last_epoch.pt"):
    ckpt = torch.load("checkpoints/last_epoch.pt", map_location="cpu")
    epoch = ckpt.get("epoch", "unknown")
    print(f"\n✓ Checkpoint found: last_epoch.pt")
    print(f"  Last saved epoch: {epoch}")
else:
    print("\n✗ No checkpoint found")

# Check training history
if os.path.exists("training_history.json"):
    with open("training_history.json", "r") as f:
        history = json.load(f)
    
    epochs = history.get("epochs", [])
    if epochs:
        last = epochs[-1]
        print(f"\n✓ Training history found")
        print(f"  Latest epoch: {last['epoch']}")
        print(f"  Train loss: {last['train_loss']:.4f}")
        print(f"  Val loss: {last['val_loss']:.4f}")
        print(f"  Train acc: {last.get('train_acc', 0):.4f}")
        print(f"  Val WER: {last['val_wer']:.4f}")
        print(f"  Val CER: {last.get('val_cer', 0):.4f}")
        
        blank_ratio = last.get('error_stats', {}).get('blank_ratio', 0)
        if blank_ratio > 0:
            print(f"  Blank ratio: {blank_ratio:.2%}")
            if blank_ratio > 0.9:
                print(f"  ⚠️  WARNING: High blank ratio - model may be collapsing!")
    else:
        print("\n✗ Training history is empty")
else:
    print("\n✗ No training history found")

# Check if training is running
print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)

if epochs and last['epoch'] < 30:
    print(f"\n⚠️  Training incomplete: Only {last['epoch']} epochs completed (target: 30)")
    print(f"   Val WER is {last['val_wer']:.4f} (target: <0.35)")
    if last['val_wer'] >= 1.0:
        print(f"   ⚠️  WER is stuck at 1.0 - model may need restart with fixes")
    print(f"\n   To continue training:")
    print(f"   python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 30 --batch_size 8 --lr 1e-3")
else:
    print("\n✓ Training appears complete or ready to start")
    print(f"   Run: python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 30 --batch_size 8 --lr 1e-3")

print("=" * 60)

