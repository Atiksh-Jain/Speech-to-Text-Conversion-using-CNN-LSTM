# How to Add 800 Recordings (Normal/Manual Way)

## Simple Process

### Step 1: Record Your Audio Files
- Record 800 audio files (WAV format preferred)
- Name them however you want (e.g., `recording_001.wav`, `sample_001.wav`, etc.)
- Save them to: `data/raw/` directory

**Example:**
```
data/raw/
├── recording_001.wav
├── recording_002.wav
├── recording_003.wav
└── ... (800 files total)
```

### Step 2: Update train.csv
Open `data/manifests/train.csv` and add entries for each file:

**CSV Format:**
```csv
path,text
data/raw/recording_001.wav,hello how are you doing today
data/raw/recording_002.wav,this is my final year project
data/raw/recording_003.wav,speech recognition using cnn lstm
... (one line per file)
```

**Important:**
- `path` column: Use `data/raw/filename.wav` (matches your current format)
- `text` column: Exact transcription of what you said in the audio
- No spaces after commas
- One line per audio file

### Step 3: Verify
```bash
python check_data_quality.py
```

---

## Example CSV Entry

If you have a file at:
```
data/raw/my_recording_001.wav
```

And you said: "hello how are you doing today"

Then in `data/manifests/train.csv`, add:
```csv
data/raw/my_recording_001.wav,hello how are you doing today
```

---

## Quick Tips

### File Naming
- Can be any name: `rec_001.wav`, `sample_001.wav`, `audio_001.wav`, etc.
- Just make sure the path in CSV matches the filename

### CSV Path Format
- ✅ Correct: `data/raw/recording_001.wav` (matches your current format)
- ✅ Correct: `data/raw/user_recorded/sample_001.wav` (if in subdirectory)
- ❌ Wrong: `raw/recording_001.wav` (doesn't match your format)
- ❌ Wrong: `C:\Users\...\recording_001.wav` (don't use absolute paths)

### Transcription Tips
- Match exactly what you said
- Use lowercase (system converts automatically)
- No special punctuation needed
- Keep it simple: "hello world" not "Hello, world!"

---

## After Adding All 800 Files

### 1. Check Data Quality
```bash
python check_data_quality.py
```
Should show:
- ✅ Total samples: ~1180 (380 existing + 800 new)
- ✅ No critical errors
- ⚠️  Warnings about sample rate/stereo are OK (auto-handled)

### 2. Test System
```bash
python test_training_dryrun.py
```
Should pass all tests.

### 3. Train Model
```bash
python run_all.py
```

---

## Current Status

**Already in train.csv:**
- 170 existing samples (your original data)
- 210 TTS samples (just generated)
- **Total: 380 samples**

**After adding 800:**
- 380 existing + 800 new = **1180 total samples** ✅

---

## That's It!

1. Record 800 audio files → Save to `data/raw/`
2. Add 800 lines to `data/manifests/train.csv` with format: `data/raw/filename.wav,transcription`
3. Run `python check_data_quality.py`
4. Run `python run_all.py`

**Simple and straightforward!** 🚀

