# Next Steps After Adding 800 Recordings

## Step-by-Step Process

### Step 1: Verify Data Quality ✅
**Run this first to check everything is correct:**
```bash
python check_data_quality.py
```

**What to look for:**
- ✅ Total samples: ~1180 (380 existing + 800 new)
- ✅ No critical errors (corrupted files)
- ⚠️  Warnings about sample rate/stereo are OK (auto-handled)

**If errors found:**
- Fix corrupted files or remove them from CSV
- Re-run the check until no errors

---

### Step 2: Test System (Optional but Recommended) ✅
**Verify the system can load your data:**
```bash
python test_training_dryrun.py
```

**What to look for:**
- ✅ All tests pass
- ✅ Data loads successfully
- ✅ Model can process samples

**If tests fail:**
- Check error messages
- Verify CSV format is correct
- Ensure all audio files are accessible

---

### Step 3: Train the Model 🚀
**This is the main step - training your model:**
```bash
python run_all.py
```

**What happens:**
- Trains for 30 epochs (2-4 hours on CPU)
- Saves checkpoints every epoch
- Saves best model based on WER
- Generates all performance plots
- Creates training history

**What to expect:**
- Training will take 2-4 hours
- You'll see progress updates
- Loss should decrease over time
- WER should improve (start high, decrease)

**Don't stop training early!** Let it complete all 30 epochs.

---

### Step 4: Monitor Training Progress (Optional) 📊
**While training is running, you can check progress:**
```bash
# In another terminal, run:
python check_training.py
```

**What to look for:**
- Loss decreasing (good sign)
- WER decreasing (good sign)
- Train accuracy increasing (good sign)

**If WER stays at 1.0:**
- Wait for more epochs (needs time to learn)
- Check after epoch 10-15
- If still 1.0 after 20 epochs, check data quality

---

### Step 5: Verify Training Completed ✅
**After training finishes, check results:**
```bash
# Check training history
python check_training.py

# Check if checkpoints exist
dir checkpoints
```

**What to look for:**
- `checkpoints/best_by_wer.pt` exists (best model)
- `checkpoints/last_epoch.pt` exists (last checkpoint)
- `training_history.json` exists (training metrics)
- Plots in `web/static/plots/` directory

---

### Step 6: Evaluate Model (Optional) 📈
**Test the trained model on validation set:**
```bash
python -m src.evaluate \
    --csv data/manifests/val.csv \
    --checkpoint checkpoints/best_by_wer.pt
```

**What to look for:**
- WER (Word Error Rate): Lower is better (0.3-0.6 is good)
- CER (Character Error Rate): Lower is better (0.2-0.4 is good)
- Transcriptions appear (not empty)

---

### Step 7: Test Inference (Optional) 🎤
**Test the model on a single audio file:**
```bash
python -m src.infer \
    --audio data/raw/utt001.wav \
    --checkpoint checkpoints/best_by_wer.pt
```

**What to look for:**
- Transcription appears (not empty)
- Text matches audio (may have some errors)

---

### Step 8: Test Web App 🌐
**Launch the web interface:**
```bash
cd web
python app.py
```

**Then:**
1. Open browser: `http://localhost:5000`
2. Click "Start Recording"
3. Speak into microphone
4. Click "Stop"
5. See transcription

**What to look for:**
- Web app loads
- Microphone access works
- Transcription appears (may have errors)
- Plots and diagrams display

---

## Quick Command Summary

```bash
# 1. Verify data
python check_data_quality.py

# 2. Test system (optional)
python test_training_dryrun.py

# 3. Train model (main step - takes 2-4 hours)
python run_all.py

# 4. Check training progress (optional, while training)
python check_training.py

# 5. Evaluate model (optional)
python -m src.evaluate --csv data/manifests/val.csv --checkpoint checkpoints/best_by_wer.pt

# 6. Test inference (optional)
python -m src.infer --audio data/raw/utt001.wav --checkpoint checkpoints/best_by_wer.pt

# 7. Test web app
cd web
python app.py
```

---

## Expected Timeline

| Step | Time | Description |
|------|------|-------------|
| 1. Verify Data | 2-5 min | Check data quality |
| 2. Test System | 1-2 min | Verify data loading |
| 3. Train Model | 2-4 hours | Main training process |
| 4. Monitor | Ongoing | Check progress |
| 5. Verify | 1 min | Check results |
| 6. Evaluate | 2-5 min | Test on validation |
| 7. Test Inference | 1 min | Test single file |
| 8. Test Web App | 5-10 min | Test web interface |

**Total: ~3-5 hours (mostly training time)**

---

## Troubleshooting

### If check_data_quality.py shows errors:
- Fix corrupted files
- Remove missing files from CSV
- Re-run check

### If test_training_dryrun.py fails:
- Check CSV format
- Verify file paths are correct
- Ensure all files are accessible

### If training fails:
- Check error message
- Verify data quality (no corrupted files)
- Check disk space
- Reduce batch size if memory issue

### If WER stays high (>0.8):
- Wait for more epochs (needs time)
- Check after epoch 20
- Verify data quality
- Check transcriptions match audio

### If transcriptions are empty:
- Model may need more training
- Check training progress
- Verify model is learning (loss decreasing)
- May need more epochs

---

## Success Criteria

### Minimum Success:
- ✅ Training completes without errors
- ✅ Checkpoints saved
- ✅ WER < 0.7 (70% word error rate)
- ✅ Transcriptions appear (not all blanks)
- ✅ Web app works

### Good Success:
- ✅ WER 0.4-0.6 (40-60% word error rate)
- ✅ Readable transcriptions
- ✅ Works for new speakers (similar to training)
- ✅ Clean demo

### Excellent Success:
- ✅ WER 0.3-0.4 (30-40% word error rate)
- ✅ High-quality transcriptions
- ✅ Robust to variations
- ✅ Production-like demo

---

## Bottom Line

**After adding 800 recordings:**

1. ✅ **Verify:** `python check_data_quality.py`
2. 🚀 **Train:** `python run_all.py` (2-4 hours)
3. ✅ **Test:** `cd web && python app.py`

**That's it! You're done!** 🎉

---

## Need Help?

- Check `HOW_TO_ADD_800_RECORDINGS.md` for adding recordings
- Check `READY_FOR_TOMORROW.md` for complete guide
- Check `HONEST_ANSWER.md` for troubleshooting
- Check `VERIFICATION_PLAN.md` for detailed verification

**You've got this!** 🚀

