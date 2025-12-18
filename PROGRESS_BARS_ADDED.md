# Progress Bars Added to Training and Validation

## ✅ What Was Added

I've added **live progress bars with ETA** to both training and validation loops using `tqdm`!

---

## Features

### 1. Training Progress Bar ✅
- Shows **real-time progress** during each epoch
- Displays:
  - Current batch number / total batches
  - Elapsed time and **ETA** (estimated time remaining)
  - Processing rate (batches/second)
  - **Live metrics**: loss, accuracy, learning rate
- Updates in real-time as training progresses

### 2. Validation Progress Bar ✅
- Shows **real-time progress** during validation
- Displays:
  - Current batch number / total batches
  - Elapsed time and **ETA**
  - Processing rate
  - **Live metrics**: loss, WER, CER
- Updates in real-time as validation progresses

### 3. Evaluation Progress Bar ✅
- Shows **real-time progress** during evaluation
- Displays:
  - Current sample number / total samples
  - Elapsed time and **ETA**
  - Processing rate
  - **Live metrics**: WER, CER, sample count

---

## Example Output

### During Training:
```
Epoch 1/30 [Train] |████████████████████| 139/139 [02:15<00:00, 1.03batch/s, loss=45.2341, acc=0.1234, lr=2.00e-04]
Epoch 1/30 [Val]   |████████████████████| 13/13 [00:12<00:00, 1.08batch/s, loss=42.1234, WER=0.4567, CER=0.3456]
```

### Epoch Summary:
```
────────────────────────────────────────────────────────────────────────────────
Epoch 1/30 Summary:
  Train: loss=45.2341, acc=0.1234
  Val:   loss=42.1234, WER=0.4567, CER=0.3456
  LR:    0.000200
  ✓ New best WER: 0.4567 (previous: inf)
────────────────────────────────────────────────────────────────────────────────
```

### During Evaluation:
```
Evaluating |████████████████████| 101/101 [01:23<00:00, 1.21sample/s, WER=0.3456, CER=0.2345, samples=101]

================================================================================
Evaluation Results:
  Total samples: 101
  WER: 0.3456
  CER: 0.2345
================================================================================
```

---

## What You'll See

### Training:
1. **Training progress bar** - Shows live training metrics
2. **Validation progress bar** - Shows live validation metrics
3. **Epoch summary** - Clean summary after each epoch

### Evaluation:
1. **Evaluation progress bar** - Shows live evaluation metrics
2. **Final results** - Summary at the end

---

## Installation

The `tqdm` library has been added to `requirements.txt`. Install it with:

```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install tqdm==4.66.1
```

---

## Usage

### Training (automatic):
```bash
python -m src.train \
    --train_csv data/manifests/train.csv \
    --val_csv data/manifests/val.csv \
    --epochs 30
```

**Progress bars appear automatically!** No configuration needed.

### Evaluation (automatic):
```bash
python -m src.evaluate \
    --csv data/manifests/val.csv \
    --checkpoint checkpoints/best_by_wer.pt
```

**Progress bar appears automatically!**

---

## Benefits

### ✅ Real-Time Feedback
- See exactly what's happening during training
- Know how long training will take (ETA)
- Monitor metrics as they update

### ✅ Better UX
- Professional-looking progress bars
- Clear visual feedback
- Easy to see if training is progressing

### ✅ Debugging
- Spot issues early (if loss isn't decreasing)
- See if training is stuck
- Monitor learning rate changes

---

## Progress Bar Information

### Training Bar Shows:
- **Progress**: `139/139` (current batch / total batches)
- **Time**: `[02:15<00:00]` (elapsed < remaining)
- **Rate**: `1.03batch/s` (processing speed)
- **Metrics**: `loss=45.23, acc=0.12, lr=2.00e-04`

### Validation Bar Shows:
- **Progress**: `13/13` (current batch / total batches)
- **Time**: `[00:12<00:00]` (elapsed < remaining)
- **Rate**: `1.08batch/s` (processing speed)
- **Metrics**: `loss=42.12, WER=0.45, CER=0.34`

### Evaluation Bar Shows:
- **Progress**: `101/101` (current sample / total samples)
- **Time**: `[01:23<00:00]` (elapsed < remaining)
- **Rate**: `1.21sample/s` (processing speed)
- **Metrics**: `WER=0.34, CER=0.23, samples=101`

---

## Technical Details

### Files Modified:
- ✅ `requirements.txt` - Added `tqdm==4.66.1`
- ✅ `src/train.py` - Added progress bars to training and validation
- ✅ `src/evaluate.py` - Added progress bar to evaluation

### Implementation:
- Uses `tqdm` library for progress bars
- Updates metrics in real-time
- Shows ETA based on current processing speed
- Clean, professional formatting

---

## Example Full Training Output

```
================================================================================
Starting Training: 30 epochs, batch_size=8, lr=0.001
Train samples: 1111, Val samples: 101
================================================================================

Epoch 1/30 [Train] |████████████████████| 139/139 [02:15<00:00, 1.03batch/s, loss=45.2341, acc=0.1234, lr=2.00e-04]
Epoch 1/30 [Val]   |████████████████████| 13/13 [00:12<00:00, 1.08batch/s, loss=42.1234, WER=0.4567, CER=0.3456]

────────────────────────────────────────────────────────────────────────────────
Epoch 1/30 Summary:
  Train: loss=45.2341, acc=0.1234
  Val:   loss=42.1234, WER=0.4567, CER=0.3456
  LR:    0.000200
  ✓ New best WER: 0.4567 (previous: inf)
────────────────────────────────────────────────────────────────────────────────

Epoch 2/30 [Train] |████████████████████| 139/139 [02:14<00:00, 1.03batch/s, loss=38.1234, acc=0.2345, lr=4.00e-04]
Epoch 2/30 [Val]   |████████████████████| 13/13 [00:12<00:00, 1.08batch/s, loss=35.2345, WER=0.3456, CER=0.2345]

────────────────────────────────────────────────────────────────────────────────
Epoch 2/30 Summary:
  Train: loss=38.1234, acc=0.2345
  Val:   loss=35.2345, WER=0.3456, CER=0.2345
  LR:    0.000400
  ✓ New best WER: 0.3456 (previous: 0.4567)
────────────────────────────────────────────────────────────────────────────────

...
```

---

## Summary

**✅ Added live progress bars with ETA to:**
- Training loop
- Validation loop
- Evaluation script

**✅ Shows:**
- Real-time progress
- ETA (estimated time remaining)
- Live metrics (loss, accuracy, WER, CER)
- Processing rate

**✅ Benefits:**
- Better user experience
- Real-time feedback
- Easy debugging
- Professional appearance

**Just run training/evaluation and you'll see the progress bars automatically!** 🎉

