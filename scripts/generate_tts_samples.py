"""
Generate TTS (Text-to-Speech) samples to augment your dataset.
Uses pyttsx3 (offline) or gTTS (online) to generate audio from text.
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import torchaudio

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    print("⚠️  pyttsx3 not available. Install with: pip install pyttsx3")

try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("⚠️  gTTS not available. Install with: pip install gtts")

# Sample sentences for TTS generation
TTS_SENTENCES = [
    "hello how are you doing today",
    "this is my final year project",
    "speech recognition using cnn lstm",
    "good morning everyone",
    "the weather is nice today",
    "i am working on my project",
    "thank you for your help",
    "can you please repeat that",
    "the system works correctly",
    "i need to test this application",
    "the microphone is working fine",
    "this is a test recording",
    "the model is learning well",
    "i hope this works properly",
    "let me try again",
    "the accuracy is improving",
    "this is sample number one",
    "the training is progressing",
    "i am recording my voice",
    "the system recognizes speech",
    "this is a voice recognition system",
    "the demo will work perfectly",
    "i am testing the microphone",
    "the audio quality is good",
    "this is my voice sample",
    "the model needs more data",
    "i am speaking clearly",
    "the transcription should be accurate",
    "this is a speech to text system",
    "the neural network is learning",
    "machine learning is fascinating",
    "deep learning models are powerful",
    "artificial intelligence is the future",
    "natural language processing is complex",
    "the computer understands my speech",
    "voice recognition technology is advanced",
    "the system processes audio signals",
    "feature extraction is important",
    "the model uses convolutional layers",
    "long short term memory networks",
    "connectionist temporal classification",
    "the model predicts character sequences",
    "training requires large datasets",
    "validation helps prevent overfitting",
    "the loss function measures accuracy",
    "gradient descent optimizes parameters",
    "backpropagation updates weights",
    "the model learns from examples",
    "data augmentation improves robustness",
    "the system handles different speakers",
    "accent variations are challenging",
    "noise reduction improves accuracy",
    "the model generalizes to new data",
    "inference speed is important",
    "real time processing is possible",
    "the web interface is user friendly",
    "the demo shows system capabilities",
    "evaluation metrics measure performance",
    "word error rate indicates accuracy",
    "character error rate shows precision",
    "the system works for multiple languages",
    "speech recognition has many applications",
    "voice assistants use similar technology",
    "the model can be improved with more data",
    "transfer learning helps with small datasets",
    "fine tuning adapts to specific domains",
    "the architecture is well designed",
    "the implementation is efficient",
    "the code is production ready",
    "documentation helps users understand",
    "testing ensures system reliability",
    "the project demonstrates machine learning",
    "deep learning requires computational resources",
    "cpu training is slower but accessible",
    "gpu acceleration speeds up training",
    "the model architecture is important",
    "hyperparameters affect performance",
    "learning rate controls training speed",
    "batch size affects memory usage",
    "epochs determine training duration",
    "early stopping prevents overfitting",
    "checkpointing saves progress",
    "the model can be deployed",
    "inference is fast and accurate",
    "the system is ready for demo",
]

def generate_with_pyttsx3(text: str, output_path: str, rate: int = 150):
    """Generate TTS using pyttsx3 (offline, no internet needed)."""
    if not PYTTSX3_AVAILABLE:
        raise ImportError("pyttsx3 not available")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    
    # Convert to WAV if needed and resample to 16kHz
    if output_path.endswith('.wav'):
        wav, sr = torchaudio.load(output_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        torchaudio.save(output_path, wav, 16000)

def generate_with_gtts(text: str, output_path: str, lang: str = 'en'):
    """Generate TTS using gTTS (requires internet)."""
    if not GTTS_AVAILABLE:
        raise ImportError("gTTS not available")
    
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Save to temporary file then convert
    temp_path = output_path + '.mp3'
    tts.save(temp_path)
    
    # Convert MP3 to WAV and resample to 16kHz
    try:
        wav, sr = torchaudio.load(temp_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        torchaudio.save(output_path, wav, 16000)
        os.remove(temp_path)  # Delete MP3 file
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def generate_tts_samples(
    output_dir: str,
    csv_path: str,
    num_samples: int = 200,
    method: str = 'pyttsx3',
    use_custom_sentences: bool = False,
    custom_sentences_file: str = None
):
    """
    Generate TTS samples and add to CSV.
    
    Args:
        output_dir: Directory to save TTS audio files
        csv_path: CSV file to update
        num_samples: Number of TTS samples to generate
        method: 'pyttsx3' (offline) or 'gtts' (online)
        use_custom_sentences: Use custom sentences from file
        custom_sentences_file: Path to file with custom sentences (one per line)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("TTS SAMPLE GENERATION")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Output directory: {output_dir}")
    print(f"   CSV file: {csv_path}")
    print(f"   Number of samples: {num_samples}")
    print(f"   Method: {method}")
    
    # Get sentences
    if use_custom_sentences and custom_sentences_file:
        print(f"\n📝 Loading custom sentences from: {custom_sentences_file}")
        with open(custom_sentences_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
    else:
        sentences = TTS_SENTENCES
    
    if len(sentences) < num_samples:
        # Repeat sentences if needed
        sentences = (sentences * ((num_samples // len(sentences)) + 1))[:num_samples]
    
    # Load or create CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        start_idx = len(df) + 1
        print(f"\n📋 Existing CSV has {len(df)} samples")
    else:
        df = pd.DataFrame(columns=['path', 'text'])
        start_idx = 1
        print(f"\n📋 Creating new CSV")
    
    print(f"\n🔄 Generating {num_samples} TTS samples...")
    
    new_samples = []
    failed = 0
    
    for i in range(num_samples):
        text = sentences[i % len(sentences)]
        
        # Generate filename
        filename = f"tts_{start_idx + i:04d}.wav"
        filepath = os.path.join(output_dir, filename)
        
        try:
            # Generate TTS
            if method == 'pyttsx3':
                if not PYTTSX3_AVAILABLE:
                    print(f"   ❌ pyttsx3 not available. Install with: pip install pyttsx3")
                    break
                generate_with_pyttsx3(text, filepath)
            elif method == 'gtts':
                if not GTTS_AVAILABLE:
                    print(f"   ❌ gTTS not available. Install with: pip install gtts")
                    break
                generate_with_gtts(text, filepath)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Add to CSV
            rel_path = os.path.relpath(filepath, os.path.dirname(csv_path))
            new_samples.append({'path': rel_path, 'text': text})
            
            if (i + 1) % 20 == 0:
                print(f"   Generated: {i + 1}/{num_samples}")
        
        except Exception as e:
            print(f"   ⚠️  Failed to generate sample {i + 1}: {e}")
            failed += 1
            continue
    
    # Update CSV
    if new_samples:
        new_df = pd.DataFrame(new_samples)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("✅ TTS GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Generated: {len(new_samples)} samples")
    print(f"   Failed: {failed} samples")
    print(f"   Total in CSV: {len(combined_df) if new_samples else len(df)} samples")
    print(f"   CSV saved: {csv_path}")
    print(f"\nNext steps:")
    print(f"1. Check data quality: python check_data_quality.py")
    print(f"2. Add your 800 recordings tomorrow")
    print(f"3. Train model: python run_all.py")

def main():
    parser = argparse.ArgumentParser(description='Generate TTS samples for dataset augmentation')
    parser.add_argument('--output_dir', type=str, default='data/raw/tts_generated/',
                       help='Directory to save TTS audio files')
    parser.add_argument('--csv_path', type=str, default='data/manifests/train.csv',
                       help='CSV file to update')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='Number of TTS samples to generate')
    parser.add_argument('--method', type=str, default='pyttsx3', choices=['pyttsx3', 'gtts'],
                       help='TTS method: pyttsx3 (offline) or gtts (online)')
    parser.add_argument('--custom_sentences', type=str, default=None,
                       help='Path to file with custom sentences (one per line)')
    
    args = parser.parse_args()
    
    generate_tts_samples(
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        num_samples=args.num_samples,
        method=args.method,
        use_custom_sentences=args.custom_sentences is not None,
        custom_sentences_file=args.custom_sentences
    )

if __name__ == "__main__":
    main()

