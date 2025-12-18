# Training Fixes Applied

## Summary

All critical fixes have been applied to resolve the blank token collapse issue. The model should now learn correctly.

---

## Fixes Applied

### 1. ✅ Fixed Model Initialization (CRITICAL)
**File**: `src/train.py` (line 193)

**Before**:
```python
torch.nn.init.xavier_uniform_(param, gain=0.1)  # Too small!
```

**After**:
```python
torch.nn.init.xavier_uniform_(param, gain=1.0)  # Proper initialization
```

**Impact**: Model now starts with proper weight scales, preventing collapse to blank predictions.

---

### 2. ✅ Lowered Learning Rate
**File**: `src/train.py` (line 171)

**Before**:
```python
parser.add_argument("--lr", type=float, default=1e-3)
```

**After**:
```python
parser.add_argument("--lr", type=float, default=1e-4)  # Lower, more stable
```

**Impact**: More stable training, reduces risk of overshooting and collapsing to blanks.

---

### 3. ✅ Improved CTC Loss Configuration
**File**: `src/train.py` (line 187)

**Before**:
```python
criterion = nn.CTCLoss(blank=0, zero_infinity=False, reduction='mean')
```

**After**:
```python
criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
```

**Impact**: Handles infinite losses gracefully, more stable training.

---

### 4. ✅ Added Blank Token Monitoring
**File**: `src/train.py` (validation function)

**New Features**:
- Tracks blank token ratio during validation
- Shows blank% in progress bar
- Warns if blank ratio > 90%
- Saves blank ratio to training history

**Impact**: Early detection of blank collapse issue during training.

---

## How to Use

### Option 1: Use the Restart Script (Recommended)

```bash
python restart_training_fixed.py
```

This script will:
- Backup existing checkpoints and history
- Start fresh training with all fixes
- Monitor for blank token issues
- Use optimal settings (50 epochs, lr=1e-4)

### Option 2: Manual Training

```bash
python -m src.train \
  --train_csv data/manifests/train.csv \
  --val_csv data/manifests/val.csv \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4
```

---

## Expected Improvements

### Before Fixes:
- ❌ WER = 1.0 (100% word error)
- ❌ CER = 1.0 (100% character error)
- ❌ All predictions empty
- ❌ Blank token probability: 77-92%

### After Fixes (Expected):
- ✅ WER should decrease over time
- ✅ CER should decrease over time
- ✅ Predictions should contain actual text
- ✅ Blank token ratio should decrease (< 50% after training)

---

## Monitoring During Training

Watch for these indicators:

### Good Signs:
- ✅ Blank% decreasing over epochs
- ✅ WER/CER decreasing
- ✅ Non-empty predictions
- ✅ Loss decreasing smoothly

### Warning Signs:
- ⚠️ Blank% > 90% after 10+ epochs
- ⚠️ WER stuck at 1.0
- ⚠️ All predictions still empty
- ⚠️ Loss not decreasing

If warnings appear, you may need to:
- Lower learning rate further (1e-5)
- Increase warmup period
- Check data quality
- Increase model capacity

---

## Files Modified

1. `src/train.py` - All fixes applied
2. `restart_training_fixed.py` - New restart script
3. `TRAINING_DIAGNOSIS.md` - Diagnostic report
4. `FIXES_APPLIED.md` - This file

---

## Next Steps

1. **Stop current training** (if still running)
2. **Run restart script**: `python restart_training_fixed.py`
3. **Monitor progress**: `python check_progress.py`
4. **Check predictions**: `python inspect_predictions.py`

---

## Technical Details

### Initialization Fix
- **Problem**: `gain=0.1` made weights too small
- **Solution**: `gain=1.0` gives proper scale
- **Why it matters**: Small weights → model can't learn → collapses to blanks

### Learning Rate Fix
- **Problem**: `1e-3` too high, causes instability
- **Solution**: `1e-4` more stable, allows gradual learning
- **Why it matters**: High LR → overshooting → collapse to local minimum (blanks)

### CTC Loss Fix
- **Problem**: `zero_infinity=False` can cause NaN/inf issues
- **Solution**: `zero_infinity=True` handles edge cases
- **Why it matters**: Prevents training crashes and instability

### Monitoring Addition
- **Problem**: No way to detect blank collapse early
- **Solution**: Track blank token ratio
- **Why it matters**: Early warning system for training issues

---

**Status**: ✅ All fixes applied and ready for training
**Date**: 2025-12-18

