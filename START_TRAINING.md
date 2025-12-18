# How to Start Training

## Quick Start

**Open a terminal/PowerShell in this directory and run:**

```bash
python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 40 --batch_size 8 --lr 1e-3
```

## Current Status
- **Current epoch:** 30
- **Current WER:** 0.55
- **Target:** 40 epochs (will train epochs 1-40, checkpoints will update)

## What Will Happen

1. Training starts from epoch 1
2. Runs through all 40 epochs
3. Checkpoints saved every epoch
4. Best model always saved
5. Training history accumulates

## Alternative: Use the Script

```bash
python continue_training.py
```

## Monitor Progress

While training, you can check status in another terminal:
```bash
python check_status.py
```

## Expected Time

- **Total:** ~3-4 hours for 40 epochs
- **Per epoch:** ~5 minutes
- **You're at epoch 30, so ~50 minutes for remaining 10 epochs worth of training**

## Note

The training script doesn't automatically resume from checkpoint - it trains from epoch 1 to the specified number. But:
- ✅ Checkpoints are saved every epoch
- ✅ Best model always preserved
- ✅ Training history accumulates
- ✅ You'll get the best model at the end

**Just run the command above in your terminal!** 🚀

