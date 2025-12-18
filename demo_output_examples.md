# What Will Happen When You Use the Website

## How It Works

1. **You open**: `http://127.0.0.1:5000/`
2. **You click**: "Start Recording" button
3. **You speak**: Any random sentence (e.g., "hello how are you doing today")
4. **You click**: "Stop Recording"
5. **Output appears**: In the "Transcription" box below

---

## Example Outputs Based on WER

### Scenario 1: WER < 0.35 (Target - Excellent!)

**You say:** "hello how are you doing today"

**Model outputs:**
- ✅ **Best case**: `"hello how are you doing today"` (perfect match!)
- ✅ **Good case**: `"hello how are you doing today"` (exact match)
- ✅ **Acceptable**: `"hello how are you doing to day"` (minor spacing)
- ✅ **Still good**: `"hello how are you doing"` (missing one word)

**What you'll see on website:**
```
Transcription:
hello how are you doing today
```

---

### Scenario 2: WER 0.35-0.50 (Good - Still Very Usable!)

**You say:** "hello how are you doing today"

**Model outputs:**
- ✅ `"hello how are you doing"` (missing last word)
- ✅ `"hello how are you doing today"` (correct!)
- ⚠️ `"hello how are you do today"` (missing "ing")
- ⚠️ `"hello how are doing today"` (missing "you")

**What you'll see on website:**
```
Transcription:
hello how are you doing
```

---

### Scenario 3: WER 0.50-0.70 (Acceptable - Some Errors)

**You say:** "hello how are you doing today"

**Model outputs:**
- ⚠️ `"hello how are you do"` (missing words)
- ⚠️ `"hello how are doing"` (missing "you")
- ⚠️ `"hello how you doing today"` (missing "are")
- ⚠️ `"hello how are doing today"` (missing "you")

**What you'll see on website:**
```
Transcription:
hello how are you do
```

---

### Scenario 4: WER > 0.70 (Needs More Training)

**You say:** "hello how are you doing today"

**Model outputs:**
- ❌ `"hello how"` (very incomplete)
- ❌ `"hello"` (only first word)
- ❌ `""` (empty - model not trained enough)
- ❌ `"how are"` (partial)

**What you'll see on website:**
```
Transcription:
hello how
```

---

## Real Examples Based on Your Current Model (WER ~0.81)

**Current Status:** Your model is at WER 0.81, so you're in Scenario 3-4 range.

**You say:** "hello how are you doing today"

**Likely output:**
- `"hello how are you"` (missing last 2 words)
- `"hello how are doing"` (missing "you")
- `"hello how are"` (missing last 3 words)

**After reaching WER < 0.35, you'll get:**
- `"hello how are you doing today"` (full sentence!)
- Or at worst: `"hello how are you doing"` (missing just one word)

---

## What the Website Interface Looks Like

```
┌─────────────────────────────────────────────────┐
│  Speech to Text using CNN–LSTM                   │
│  VTU Final Year Project – Live CPU-only ASR Demo │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│  Live Microphone Recognition                     │
│                                                  │
│  [Start Recording]  [Stop]  Status: Idle        │
│                                                  │
│  Transcription:                                  │
│  ┌─────────────────────────────────────────┐    │
│  │ hello how are you doing today           │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## How the Output is Generated

1. **Audio captured** from your microphone (16 kHz, mono)
2. **Converted to features** (Log-Mel spectrograms)
3. **Model processes** the features
4. **Beam search decoding** (beam_width=5) converts to text
5. **Text displayed** in the transcription box

---

## What Makes Output Better/Worse

### ✅ **Good Output** (WER < 0.35):
- Clear speech
- Normal speaking pace
- Common words/phrases
- Good microphone quality
- Quiet environment

### ⚠️ **Worse Output** (Higher errors):
- Fast speech
- Uncommon words
- Background noise
- Accent variations
- Very short utterances (< 0.5 seconds)

---

## Testing Your Model

After training completes, you can test with:

```bash
# Test with a file
python -m src.infer --checkpoint checkpoints/best_by_wer.pt --audio_path data/raw/utt001.wav

# Start web interface
python -m web.app
# Then open http://127.0.0.1:5000/
```

---

## Expected Behavior at Different WER Levels

| WER Range | Example Output Quality | Usability |
|-----------|----------------------|-----------|
| < 0.20    | Near-perfect          | Excellent |
| 0.20-0.35 | Very good, minor errors | Very good |
| 0.35-0.50 | Good, some missing words | Good |
| 0.50-0.70 | Acceptable, partial sentences | Acceptable |
| > 0.70    | Poor, many errors     | Needs work |

---

## Bottom Line

**After reaching WER < 0.35:**
- ✅ You'll get **complete or nearly complete sentences**
- ✅ Most words will be correct
- ✅ Output will be **readable and understandable**
- ✅ Perfect for **demonstration and VTU project**

**Current status (WER ~0.81):**
- ⚠️ You'll get **partial sentences** (missing some words)
- ⚠️ But model is **learning and improving**
- ✅ Continue training to reach target!

