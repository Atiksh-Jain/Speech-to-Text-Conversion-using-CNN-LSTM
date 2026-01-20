# Speech to Text using CNN–LSTM

This repository contains a **complete, CPU-only Speech-to-Text (ASR) system** using a **CNN–LSTM acoustic model with CTC loss**, plus a **Flask web interface** for live microphone-based transcription and visualization of training plots and system diagrams. It is designed as a **VTU final year project** and is ready for submission and demo.

The system is designed to be:

- **Production-quality**: consistent feature pipeline, robust error handling, and clear separation of modules.
- **VTU-evaluation ready**: generates all required performance graphs and static architecture diagrams.
- **Windows + CPU friendly**: no GPUs, no transformers, no HuggingFace, no wav2vec2.

---

### 1. Project Structure

The project is organized as:

```text
stt_cnn_lstm/
├── data/
│   ├── raw/
│   └── manifests/
│       ├── train.csv
│       └── val.csv
├── src/
│   ├── features.py        # Log-Mel extraction (shared everywhere)
│   ├── dataset.py         # Dataset + dataloader
│   ├── model.py           # CNN-LSTM-CTC model
│   ├── decode.py          # Greedy CTC decoding
│   ├── utils.py           # helpers, text processing, metrics
│   ├── train.py           # training loop + checkpointing + history
│   ├── evaluate.py        # WER, CER, metrics computation
│   ├── infer.py           # offline inference script
│   ├── plots.py           # automatic graph generation
├── web/
│   ├── app.py             # Flask backend
│   ├── templates/
│   │   └── index.html     # front-end UI with microphone recording
│   └── static/
│       ├── plots/         # auto-generated figures
│       └── diagrams/      # system & architecture diagrams
├── checkpoints/
│   ├── last_epoch.pt
│   └── best_by_wer.pt
└── requirements.txt
```

---

### 2. Installation

1. **Create and activate a virtual environment** (recommended):

```bash
cd stt_cnn_lstm
python -m venv venv
venv\Scripts\activate  # Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

Ensure you are using **Python 3.9+** on **Windows**. All training and inference run on **CPU**.

---

### 3. Preparing the Dataset

1. Place your **16 kHz mono WAV** files into `data/raw/`.

2. Create CSV manifest files in `data/manifests/`:

   - `train.csv`
   - `val.csv`

   Each CSV must have **two columns with headers**:

   ```text
   path,text
   data/raw/example1.wav,hello how are you
   data/raw/example2.wav,i am fine thank you
   ```

   - `path` is the relative path to the WAV file (from project root).
   - `text` is the transcription in lowercase letters, spaces, and basic punctuation.

3. The code will:
   - Safely load audio and **skip corrupt/too-short files** without crashing.
   - Perform **character-level tokenization**, reserving index `0` for the CTC blank token.

---

### 4. Training the CNN–LSTM–CTC Model

From the `stt_cnn_lstm` directory:

```bash
python -m src.train \
  --train_csv data/manifests/train.csv \
  --val_csv data/manifests/val.csv \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-3
```

Key properties:

- **Feature extraction**: 80-dim Log-Mel spectrograms (16 kHz, `n_fft=400`, `hop_length=160`, Hann window, log scaling + per-utterance mean-variance normalization).
- **Model**:
  - 2D CNN (2 Conv2D + BatchNorm + ReLU) that preserves the time dimension.
  - 2-layer **BiLSTM** with hidden size 256 and dropout 0.3.
  - Linear projection to vocabulary size + `LogSoftmax`.
  - Loss: **CTCLoss(blank=0, zero_infinity=True)**.
- **Optimizer**: Adam, `lr = 1e-3`.
- **Batch size**: 8 (CPU-safe).

During training:

- **Checkpoints are saved every epoch**:
  - `checkpoints/last_epoch.pt`
  - `checkpoints/best_by_wer.pt` (lowest validation WER).
- Training never crashes due to validation failure (bad samples are skipped gracefully).
- A `training_history.json` file is created with:
  - training loss, training accuracy
  - validation loss
  - WER, CER and error distributions per epoch

---

### 5. Evaluation & Metrics

To evaluate an existing model and compute WER/CER on the validation set:

```bash
python -m src.evaluate \
  --csv data/manifests/val.csv \
  --checkpoint checkpoints/best_by_wer.pt
```

This will:

- Load the model and compute:
  - **Word Error Rate (WER)**
  - **Character Error Rate (CER)**
  - Error distribution (substitutions, insertions, deletions)
- Print metrics to the console.

The same feature extraction and decoding logic is used as in training.

---

### 6. Offline Inference (Batch WAV Files)

You can run offline inference over one or more WAV files:

```bash
python -m src.infer \
  --checkpoint checkpoints/best_by_wer.pt \
  --audio_path data/raw/example1.wav
```

or for multiple files:

```bash
python -m src.infer \
  --checkpoint checkpoints/best_by_wer.pt \
  --audio_path data/raw/example1.wav data/raw/example2.wav
```

The script will:

- Load each WAV file.
- Apply the **same Log-Mel feature extractor** as during training.
- Run the CNN–LSTM model.
- Apply **greedy CTC decoding**.
- Print the predicted text.

Expected demo input:

- Spoken: `hello how are you doing today`
- Model output (approximate): `hello how are you doing today`

Small variations are acceptable.

---

### 7. Automatic Plot Generation

After (or during) training, you can generate all required plots for the VTU report using:

```bash
python -m src.plots
```

This reads `training_history.json` and dataset statistics and creates the following figures in `web/static/plots/`:

- **Fig. 5.1** Dataset distribution
- **Fig. 5.2** Feature extraction pipeline
- **Fig. 6.1** Log-Mel feature map visualization
- **Fig. 6.2** Training accuracy vs epochs
- **Fig. 6.3** Training vs validation loss
- **Fig. 6.4** Accuracy vs SNR (synthetic analysis)
- **Fig. 6.6** WER comparison
- **Fig. 6.7** CER analysis
- **Fig. 6.9** Performance vs baseline ASR (synthetic baseline)
- **Fig. 6.10** Error distribution
- **Fig. 2.1** ASR performance in noisy environments (synthetic)
- **Fig. 2.2** Feature comparison matrix
- **Fig. 2.3** CNN–LSTM vs Transformer comparison

These figures are simple, clear and publication-ready, and are automatically visible on the web UI.

---

### 8. Static System Diagrams

Run:

```bash
python -m src.plots --diagrams_only
```

or just rely on the default `python -m src.plots`, which also generates diagrams.

The following static diagrams are generated into `web/static/diagrams/`:

- **Fig. 4.1** Waterfall development model
- **Fig. 4.2** Overall system architecture
- **Fig. 4.3** Data flow diagram
- **Fig. 4.4** CNN–LSTM block diagram
- **Fig. 4.5** Project folder hierarchy

All diagrams are created with matplotlib as PNG images.

---

### 9. Running the Web Demo (Flask + Microphone)

1. Ensure you have at least one trained checkpoint, preferably:
   - `checkpoints/best_by_wer.pt`

2. Start the Flask server:

```bash
python -m web.app
```

3. Open your browser and navigate to:

```text
http://127.0.0.1:5000/
```

4. On the web page:

- Click **"Start Recording"**.
- Speak your utterance (e.g., *"hello how are you doing today"*).
- Click **"Stop Recording"**.
- The browser sends your recorded audio (WAV) to the Flask backend.
- The backend:
  - runs the **same feature extraction** and model inference pipeline,
  - performs greedy CTC decoding,
  - returns the **transcription**.
- The page will:
  - Display the **recognized text**.
  - Show all available **plots** from `web/static/plots/`.
  - Show all **system diagrams** from `web/static/diagrams/`.

---

### 10. Design & Safety Notes

- **Single source of truth for features**: `src/features.py` is used by:
  - Training (`train.py`)
  - Validation (`train.py`, `evaluate.py`)
  - Offline inference (`infer.py`)
  - Web inference (`web/app.py`)
- **Robust error handling**:
  - Audio loading wrapped in try/except.
  - Corrupt or too-short audio files are skipped with a clear warning.
  - All divisions guard against divide-by-zero.
  - Indexing is protected with bounds checks.
- **No disallowed dependencies**:
  - No transformers, no HuggingFace, no wav2vec2, no GPUs, no multiprocessing, no feature caching.

This repository is ready for **VTU final year project submission**, technical viva, and basic industry demonstrations.


