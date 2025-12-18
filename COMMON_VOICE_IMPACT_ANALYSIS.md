# Common Voice Impact Analysis: Training Time & Accuracy

## Current Status

### Current Dataset:
- **Train samples:** 1,110
- **Val samples:** 99
- **Batches per epoch:** ~139 (with batch_size=8)
- **Time per epoch:** ~3.5-4 minutes
  - Training: ~3.5 min
  - Validation: ~0.5 min

### Current Training Time (30 epochs):
- **Total time:** ~2-2.5 hours
- **Per epoch:** ~4 minutes average

---

## With Common Voice Added

### Option 1: Add 500 Common Voice Samples (Recommended)

#### Dataset Size:
- **Train samples:** 1,110 + 500 = **1,610 total**
- **Increase:** +45% more data
- **Batches per epoch:** ~201 (with batch_size=8)

#### Training Time:
- **Time per epoch:** ~5-5.5 minutes
  - Training: ~5 min (more batches)
  - Validation: ~0.5 min (same)
- **Total time (30 epochs):** ~2.5-3 hours
- **Time increase:** +30-60 minutes

#### Expected Accuracy:
- **Current (with improvements):** WER 0.2-0.35 (20-35% error)
- **With +500 Common Voice:** WER 0.15-0.28 (15-28% error)
- **Improvement:** **10-20% WER reduction**
- **Accuracy:** 80-90% (vs 75-85% before)

---

### Option 2: Add 1000 Common Voice Samples (Maximum Impact)

#### Dataset Size:
- **Train samples:** 1,110 + 1,000 = **2,110 total**
- **Increase:** +90% more data (almost double!)
- **Batches per epoch:** ~264 (with batch_size=8)

#### Training Time:
- **Time per epoch:** ~6.5-7 minutes
  - Training: ~6.5 min (many more batches)
  - Validation: ~0.5 min (same)
- **Total time (30 epochs):** ~3-3.5 hours
- **Time increase:** +1-1.5 hours

#### Expected Accuracy:
- **Current (with improvements):** WER 0.2-0.35 (20-35% error)
- **With +1000 Common Voice:** WER 0.12-0.25 (12-25% error)
- **Improvement:** **15-25% WER reduction**
- **Accuracy:** 85-92% (vs 75-85% before)

---

## Comparison Table

| Metric | Current | +500 CV | +1000 CV |
|--------|---------|---------|----------|
| **Train Samples** | 1,110 | 1,610 | 2,110 |
| **Batches/Epoch** | 139 | 201 | 264 |
| **Time/Epoch** | ~4 min | ~5.5 min | ~7 min |
| **Total Time (30 epochs)** | 2-2.5 hrs | 2.5-3 hrs | 3-3.5 hrs |
| **WER** | 0.2-0.35 | 0.15-0.28 | 0.12-0.25 |
| **CER** | 0.15-0.3 | 0.1-0.22 | 0.08-0.18 |
| **Accuracy** | 75-85% | 80-90% | 85-92% |
| **Improvement** | Baseline | +10-20% | +15-25% |

---

## Detailed Breakdown

### Training Time Calculation:

**Current:**
- 1,110 samples ÷ 8 batch_size = 139 batches
- 139 batches × 1.5s/batch = ~3.5 min training
- +0.5 min validation = **~4 min/epoch**
- 30 epochs × 4 min = **120 min = 2 hours**

**With +500 Common Voice:**
- 1,610 samples ÷ 8 batch_size = 201 batches
- 201 batches × 1.5s/batch = ~5 min training
- +0.5 min validation = **~5.5 min/epoch**
- 30 epochs × 5.5 min = **165 min = 2.75 hours**

**With +1000 Common Voice:**
- 2,110 samples ÷ 8 batch_size = 264 batches
- 264 batches × 1.5s/batch = ~6.5 min training
- +0.5 min validation = **~7 min/epoch**
- 30 epochs × 7 min = **210 min = 3.5 hours**

---

## Accuracy Improvement Breakdown

### Why Common Voice Helps:

1. **More Data = Better Generalization**
   - More diverse speakers
   - More diverse accents
   - More diverse speaking styles
   - Better robustness

2. **Current Limitations:**
   - Your data: Mostly your voice (limited diversity)
   - Common Voice: Thousands of speakers (high diversity)

3. **Expected Improvements:**
   - **+500 samples:** Moderate improvement (10-20% WER reduction)
   - **+1000 samples:** Significant improvement (15-25% WER reduction)

---

## Recommendations

### Option 1: Add 500 Samples (Balanced) ⭐ Recommended
- **Time:** +30-60 minutes (acceptable)
- **Accuracy:** Good improvement (10-20%)
- **Best for:** Quick improvement without too much extra time

### Option 2: Add 1000 Samples (Maximum)
- **Time:** +1-1.5 hours (longer but worth it)
- **Accuracy:** Best improvement (15-25%)
- **Best for:** Maximum accuracy if you have time

### Option 3: Skip Common Voice
- **Time:** Current 2-2.5 hours
- **Accuracy:** Still good (75-85%) with current improvements
- **Best for:** If you're short on time

---

## Final Recommendations

### If You Have Time:
**Add 1000 Common Voice samples**
- Total training: 3-3.5 hours
- Expected WER: 0.12-0.25 (excellent!)
- Accuracy: 85-92% (very good!)

### If You're Short on Time:
**Add 500 Common Voice samples**
- Total training: 2.5-3 hours
- Expected WER: 0.15-0.28 (very good!)
- Accuracy: 80-90% (good!)

### If You Want to Start Now:
**Continue current training**
- Total training: 2-2.5 hours
- Expected WER: 0.2-0.35 (good!)
- Accuracy: 75-85% (acceptable!)
- You can always add Common Voice later

---

## Summary

### With 500 Common Voice Samples:
- **Training time:** 2.5-3 hours (+30-60 min)
- **WER:** 0.15-0.28 (15-28% error) ⬇️ **10-20% improvement**
- **Accuracy:** 80-90% ⬆️ **+5-10% improvement**

### With 1000 Common Voice Samples:
- **Training time:** 3-3.5 hours (+1-1.5 hrs)
- **WER:** 0.12-0.25 (12-25% error) ⬇️ **15-25% improvement**
- **Accuracy:** 85-92% ⬆️ **+10-15% improvement**

**The extra time is worth it for the accuracy gain!** 🚀

