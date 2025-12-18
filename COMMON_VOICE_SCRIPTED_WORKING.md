# Will Common Voice Scripted Speech Work? ✅ YES!

## Answer: **YES, it will work!** ✅

The "Common Voice Scripted Speech 24.0" dataset you're looking at **will work** with your integration script.

---

## What You're Looking At

- **Dataset:** Common Voice Scripted Speech 24.0 - English
- **Format:** MP3 ✅ (script handles this)
- **Size:** 87.74 GB (HUGE - but you don't need all of it!)
- **File:** `mcv-scripted-en-v24.0.tar.gz`
- **License:** CC0-1.0 ✅ (free to use)

---

## Important Notes

### ⚠️ Size Warning:
- **87.74 GB is VERY LARGE!**
- You only need **500-1000 samples** (maybe 100-200 MB)
- **Don't download the whole thing!**

### ✅ What Will Work:
- Script handles MP3 files ✅
- Script converts to WAV automatically ✅
- Script filters by duration ✅
- Script extracts text from TSV ✅

---

## How to Use It (Recommended Approach)

### Option 1: Download and Extract Only What You Need ⭐

1. **Download the dataset** (87.74 GB - will take time)
2. **Extract it** to a folder
3. **Use the script** - it will only process what you specify:
   ```bash
   python scripts/integrate_commonvoice.py \
       --tsv_path commonvoice_data/validated.tsv \
       --audio_dir commonvoice_data/clips/ \
       --output_dir data/raw/commonvoice/ \
       --max_samples 500 \
       --target_csv data/manifests/train.csv
   ```
   The `--max_samples 500` means it will only process 500 files, not all 87GB!

### Option 2: Use Regular Common Voice (Smaller) ⭐ Recommended

Instead of Scripted Speech, consider:
- **Regular Common Voice dataset** (smaller, more diverse)
- Usually has multiple smaller files
- Easier to download and manage

**Where to find:** https://commonvoice.mozilla.org/en/datasets
- Look for "Common Voice" (not "Scripted Speech")
- Usually has smaller download options

---

## What the Script Expects

Your script looks for:
- ✅ TSV file with columns: `path`, `sentence`
- ✅ MP3 audio files in a `clips/` directory
- ✅ Audio files referenced in TSV

**Scripted Speech should have the same structure!**

---

## Step-by-Step: Using Scripted Speech

### 1. Download (if you want to):
- Click "Download" button
- Wait for 87.74 GB download (this will take hours!)
- Extract the `.tar.gz` file

### 2. Check Structure:
After extraction, you should have:
```
commonvoice_data/
  ├── clips/
  │   ├── audio_file_001.mp3
  │   └── ...
  └── validated.tsv (or train.tsv)
```

### 3. Run Integration:
```bash
python scripts/integrate_commonvoice.py \
    --tsv_path commonvoice_data/validated.tsv \
    --audio_dir commonvoice_data/clips/ \
    --output_dir data/raw/commonvoice/ \
    --max_samples 500 \
    --target_csv data/manifests/train.csv
```

**The script will:**
- ✅ Only process 500 samples (not all 87GB!)
- ✅ Convert MP3 to WAV
- ✅ Filter by duration (5-16 seconds)
- ✅ Add to your training CSV

---

## Recommendation

### ⚠️ Don't Download 87GB Just for 500 Samples!

**Better Options:**

1. **Use Regular Common Voice** (smaller files available)
   - Visit: https://commonvoice.mozilla.org/en/datasets
   - Look for smaller download options
   - Usually has validated.tsv + clips in smaller chunks

2. **Or Download Scripted Speech BUT:**
   - Only extract what you need
   - Use `--max_samples 500` to limit processing
   - You don't need to process all 87GB!

3. **Or Skip Common Voice for Now:**
   - Start training with your current 1,110 samples
   - Add Common Voice later if needed
   - Current improvements (beam search + augmentation) already help a lot!

---

## Will It Work? Final Answer

**YES, it will work!** ✅

But:
- ⚠️ 87GB is huge - consider regular Common Voice instead
- ✅ Script will only process what you specify (500-1000 samples)
- ✅ You don't need to download/process all 87GB
- ✅ Script handles MP3 conversion automatically

---

## My Recommendation

**Skip the 87GB download for now!**

Instead:
1. **Start training** with your current 1,110 samples + improvements
2. **Get results** in 2-2.5 hours
3. **If you need more accuracy later**, then download Common Voice

**OR** if you really want Common Voice:
- Look for **regular Common Voice** (not Scripted Speech)
- Usually has smaller, more manageable downloads
- Same quality, easier to work with

---

## Bottom Line

✅ **Yes, Scripted Speech will work**
⚠️ **But 87GB is overkill for 500 samples**
💡 **Better: Use regular Common Voice or skip for now**

**Your current setup (1,110 samples + improvements) is already good!** 🚀

