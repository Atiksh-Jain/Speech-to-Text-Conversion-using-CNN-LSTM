# Training Diagnosis Report

## Executive Summary

**Status**: ⚠️ Training is running but model is NOT learning correctly

**Critical Issue**: Model predicts **blank tokens at ALL timesteps** during validation, resulting in:
- WER = 1.0 (100% word error rate)
- CER = 1.0 (100% character error rate)  
- Empty predictions for all validation samples

---

## Current Training Status

✅ **Training Process**: Running (PID 25260, started 00:53:01)
- CPU time: 1186 seconds
- Memory: 1456 MB

✅ **Training Metrics (Latest Epoch 15)**:
- Train Loss: 1.93 (decreasing from 2.58) ✅
- Val Loss: 2.01 (decreasing from 2.54) ✅
- Train Accuracy: 85-87% ✅
- **WER: 1.0** ❌ (stuck)
- **CER: 0.91** ❌ (still very high)

---

## Root Cause Analysis

### Problem Identified

The model is predicting **blank token (index 0) at 100% of timesteps** during inference:

```
Blank token probability: 0.77 - 0.92 (very high)
Non-blank probabilities: 0.01 - 0.025 (very low)
Result: All predictions collapse to empty strings
```

### Why This Happens

1. **Model is stuck in local minimum**: The model learned that predicting blanks minimizes loss in early training, and never learned to predict actual characters.

2. **Training vs Validation Mismatch**: 
   - Training accuracy shows 85% (suggesting some learning)
   - But validation shows 0% (all blanks)
   - This suggests the model might be learning something during training, but not generalizing

3. **Possible Causes**:
   - Model initialization too conservative (gain=0.1 in xavier_uniform)
   - Learning rate might be too high initially, causing model to collapse to blank predictions
   - CTC loss might need different configuration
   - Model needs more training epochs
   - Feature extraction might differ between train/val

---

## Evidence

### Debug Output
```
Model output log_probs shape: (438, 1, 33)
Blank token probabilities: [0.77, 0.89, 0.92, ...] (very high)
Max non-blank probabilities: [0.026, 0.014, 0.012, ...] (very low)
Timesteps where blank wins: 438/438 (100%)
Unique predicted indices: [0] (only blank)
All blank (0)? True
```

### Sample Predictions
```
Reference:  'the model converts spoken audio into readable text'
Prediction: '' (empty)
Match: False

All 20 validation samples: Empty predictions
```

---

## Recommendations

### Immediate Actions

1. **Stop Current Training** (if still running)
   - The model is not learning correctly
   - Continuing will waste time

2. **Fix Model Initialization**
   - Change `gain=0.1` to `gain=1.0` or remove gain parameter
   - This will give model better starting point

3. **Adjust Learning Rate Schedule**
   - Current: Warmup for 5 epochs, then ReduceLROnPlateau
   - Consider: Lower initial LR (1e-4 instead of 1e-3)
   - Or: Use cosine annealing instead

4. **Add Blank Token Penalty** (optional)
   - Modify CTC loss to penalize excessive blank predictions
   - Or: Use label smoothing to prevent blank token dominance

5. **Check Feature Extraction Consistency**
   - Verify train/val use same feature extraction
   - Check normalization is consistent

### Long-term Solutions

1. **Increase Training Epochs**
   - Current: 15 epochs (may need 50-100+)
   - Add early stopping based on CER, not just WER

2. **Monitor Blank Token Ratio**
   - Add metric to track % of blank predictions
   - Should decrease over time

3. **Use Curriculum Learning**
   - Start with shorter sequences
   - Gradually increase complexity

4. **Consider Model Architecture Changes**
   - Add attention mechanism
   - Increase model capacity
   - Add residual connections

---

## Next Steps

1. ✅ Diagnose issue (DONE)
2. ⏳ Fix model initialization
3. ⏳ Adjust learning rate
4. ⏳ Restart training with fixes
5. ⏳ Monitor blank token predictions
6. ⏳ Verify predictions improve

---

## Files Modified/Created

- `check_progress.py` - Training progress analyzer
- `inspect_predictions.py` - Prediction inspector
- `debug_model_output.py` - Model output debugger
- `TRAINING_DIAGNOSIS.md` - This report

---

## Technical Details

### Model Architecture
- CNN Encoder: 2 Conv2D layers (64 channels)
- LSTM: 2-layer bidirectional (256 hidden, dropout 0.3)
- Output: Linear → LogSoftmax → CTC

### Training Configuration
- Optimizer: Adam (lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)
- Loss: CTCLoss(blank=0, zero_infinity=False)
- Batch size: 8
- Warmup: 5 epochs (linear from 0 to lr)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

### Dataset
- Train: 1111 samples
- Val: 99 samples
- Features: 80-dim Log-Mel spectrograms
- Vocab: 33 characters (blank + 32 chars)

---

**Generated**: 2025-12-18
**Status**: Needs immediate attention

