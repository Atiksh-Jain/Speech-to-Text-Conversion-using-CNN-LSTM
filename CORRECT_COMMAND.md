# ✅ CORRECT Training Command

## ❌ Common Mistakes

1. **Wrong Directory**: Running from `vtu-vtu` instead of `stt_cnn_lstm`
2. **Typo**: Using `--batch size` instead of `--batch_size`

---

## ✅ CORRECT Command

### Step 1: Navigate to the correct directory

```bash
cd stt_cnn_lstm
```

### Step 2: Run the training command

```bash
python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 350 --batch_size 8 --lr 1e-3
```

**Important:** Use `--batch_size` (with underscore), NOT `--batch size` (with space)

---

## 🚀 Quick Start (Windows)

**Option 1: Double-click the batch file**
- Double-click `start_training_fixed.bat`

**Option 2: Run from PowerShell**
```powershell
cd C:\Users\Lenovo\Desktop\vtu-vtu\stt_cnn_lstm
python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 350 --batch_size 8 --lr 1e-3
```

---

## ✅ What Should Happen

```
================================================================================
Loading EVERYTHING from BEST checkpoint (best_by_wer.pt)
Resuming from epoch 50 with best model state
================================================================================
[OK] Best model weights loaded
[OK] Optimizer state loaded (from best checkpoint)
[OK] Starting from epoch 50 (as requested)
[OK] Best WER: 0.4914
[OK] No improve count reset to 0 (fresh start from epoch 50)
```

---

## 📋 Copy-Paste Ready Command

```bash
cd stt_cnn_lstm && python -m src.train --train_csv data/manifests/train.csv --val_csv data/manifests/val.csv --epochs 350 --batch_size 8 --lr 1e-3
```

---

**Make sure you're in the `stt_cnn_lstm` directory before running!** 🎯

