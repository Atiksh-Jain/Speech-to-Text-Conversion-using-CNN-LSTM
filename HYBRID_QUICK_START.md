# Hybrid Strategy Quick Start Guide

## What We've Added (Best of Both Worlds)

### ✅ Enhanced Recording Script
- Structured prompts (like Mimic Recording Studio)
- Automatic CSV generation
- Better organization

### ✅ Common Voice Integration
- Download and process Common Voice dataset
- Combine with your data
- Still train from scratch (more impressive!)

### ✅ Model Optimization
- Optimize trained models for production
- Faster inference
- Production-ready checkpoints

---

## Quick Start: Hybrid Data Collection

### Option 1: Your Data + Common Voice (Recommended)

**Step 1: Record Your Own Samples**
```bash
# Record 400-500 samples with structured prompts
python scripts/record_with_prompts.py \
    --output_dir data/raw/user_recorded/ \
    --csv_path data/manifests/train.csv \
    --num_samples 400 \
    --duration 5.0
```

**Step 2: Add Common Voice Data**
```bash
# First, download Common Voice from:
# https://commonvoice.mozilla.org/

# Then integrate it:
python scripts/integrate_commonvoice.py \
    --tsv_path commonvoice_data/validated.tsv \
    --audio_dir commonvoice_data/clips/ \
    --output_dir data/raw/commonvoice/ \
    --max_samples 500 \
    --target_csv data/manifests/train.csv
```

**Step 3: Train (Still From Scratch!)**
```bash
python run_all.py
```

**Result:**
- 400-500 your samples
- 500 Common Voice samples
- Total: 900-1000 samples
- Still training from scratch (impressive!)
- Better accuracy (more diverse data)

---

### Option 2: Your Data + TTS (Current Plan)

**Step 1: Record Your Samples**
```bash
python scripts/record_with_prompts.py \
    --output_dir data/raw/user_recorded/ \
    --csv_path data/manifests/train.csv \
    --num_samples 600 \
    --duration 5.0
```

**Step 2: Generate TTS Samples**
(Use your existing TTS generation method)

**Step 3: Train**
```bash
python run_all.py
```

---

## After Training: Optimize Model

**Step 1: Train Your Model**
```bash
python run_all.py
```

**Step 2: Optimize for Production**
```bash
python -m src.optimize \
    --checkpoint checkpoints/best_by_wer.pt \
    --output checkpoints/optimized_model.pt
```

**Step 3: Use Optimized Model**
- Update `web/app.py` to use optimized checkpoint
- Faster inference
- Production-ready

---

## Benefits of Hybrid Approach

### ✅ Maintains Impressiveness
- Still training from scratch
- No pretrained model dependencies
- Full control over process

### ✅ Improves Results
- More diverse data (Common Voice)
- Better generalization
- Higher accuracy

### ✅ Adds Best Practices
- Structured data collection
- Model optimization
- Production-ready outputs

### ✅ Still VTU-Appropriate
- Demonstrates data sourcing skills
- Shows understanding of datasets
- Combines multiple sources intelligently

---

## Recommended Data Strategy

### Best Approach (Hybrid):
```
400-500 your recordings
+ 500 Common Voice samples
+ 200-300 TTS samples (optional)
= 1100-1300 total samples
```

**Training:**
- Train from scratch on combined dataset
- More impressive (using multiple sources)
- Better results (more diverse data)

---

## Quick Commands Reference

### Recording
```bash
# Record with prompts
python scripts/record_with_prompts.py --num_samples 400

# Check instructions
python scripts/integrate_commonvoice.py --instructions
```

### Data Quality
```bash
# Check data quality
python check_data_quality.py

# List corrupted files
python list_corrupted_files.py
```

### Training
```bash
# Test system first
python test_training_dryrun.py

# Train model
python run_all.py

# Check training progress
python check_training.py
```

### Optimization
```bash
# Optimize trained model
python -m src.optimize \
    --checkpoint checkpoints/best_by_wer.pt \
    --output checkpoints/optimized_model.pt
```

---

## What to Tell Your Examiner

**If Asked About Data Sources:**
- "Used multiple data sources for robustness"
- "Combined my own recordings with Common Voice dataset"
- "Still trained from scratch (no pretrained models)"
- "Demonstrates ability to work with real-world datasets"

**If Asked About Approach:**
- "Combined best practices from research"
- "Structured data collection for consistency"
- "Model optimization for production deployment"
- "Full control over entire pipeline"

**This shows:**
- Data sourcing skills ✅
- Understanding of datasets ✅
- Production considerations ✅
- Still impressive (from scratch) ✅

---

## Bottom Line

**Hybrid Strategy = Best of Both Worlds**

✅ Keep: Training from scratch (impressive)
✅ Add: Common Voice data (more data, better results)
✅ Add: Structured recording (better organization)
✅ Add: Model optimization (production-ready)

**Result:**
- Still impressive for VTU
- Better accuracy (more diverse data)
- Better organization
- Production-ready system

**This is the perfect balance!** 🚀

