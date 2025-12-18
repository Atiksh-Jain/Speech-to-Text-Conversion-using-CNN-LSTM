"""
Script to integrate Common Voice dataset with your existing data.
Downloads, processes, and combines Common Voice samples with your data.
Still trains from scratch (combines all data sources).
"""

import os
import pandas as pd
import argparse
from pathlib import Path
import torchaudio
from typing import List, Tuple

def download_commonvoice_instructions():
    """Print instructions for downloading Common Voice."""
    print("=" * 70)
    print("COMMON VOICE DATASET INTEGRATION")
    print("=" * 70)
    print("\nStep 1: Download Common Voice Dataset")
    print("   Visit: https://commonvoice.mozilla.org/")
    print("   Download English dataset (any version)")
    print("   You'll get a .tsv file and audio files")
    print("\nStep 2: Extract the dataset")
    print("   Extract to a folder (e.g., commonvoice_data/)")
    print("   Structure should be:")
    print("     commonvoice_data/")
    print("       ├── clips/")
    print("       │   ├── common_voice_en_0001.mp3")
    print("       │   └── ...")
    print("       └── validated.tsv (or train.tsv)")
    print("\nStep 3: Run this script:")
    print("   python scripts/integrate_commonvoice.py \\")
    print("       --tsv_path commonvoice_data/validated.tsv \\")
    print("       --audio_dir commonvoice_data/clips/ \\")
    print("       --output_dir data/raw/commonvoice/ \\")
    print("       --max_samples 500 \\")
    print("       --target_csv data/manifests/train.csv")
    print("\nThis will:")
    print("  - Convert MP3 to WAV")
    print("  - Filter by duration (5-16 seconds)")
    print("  - Add to your training CSV")
    print("  - Still train from scratch (combined dataset)")
    print("=" * 70)

def convert_mp3_to_wav(mp3_path: str, wav_path: str) -> bool:
    """Convert MP3 to WAV format."""
    try:
        wav, sr = torchaudio.load(mp3_path)
        # Resample to 16kHz if needed
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        # Convert to mono if stereo
        if wav.dim() > 1 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # Save as WAV
        torchaudio.save(wav_path, wav, 16000)
        return True
    except Exception as e:
        print(f"  ⚠️  Failed to convert {mp3_path}: {e}")
        return False

def process_commonvoice(
    tsv_path: str,
    audio_dir: str,
    output_dir: str,
    max_samples: int = 500,
    min_duration: float = 5.0,
    max_duration: float = 16.0
) -> List[Tuple[str, str]]:
    """
    Process Common Voice dataset and return list of (path, text) tuples.
    
    Args:
        tsv_path: Path to Common Voice TSV file
        audio_dir: Directory containing audio files
        output_dir: Directory to save converted WAV files
        max_samples: Maximum number of samples to process
        min_duration: Minimum audio duration in seconds
        max_duration: Maximum audio duration in seconds
    
    Returns:
        List of (wav_path, text) tuples
    """
    print(f"\n📋 Reading Common Voice TSV: {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    print(f"   Total samples in TSV: {len(df)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    samples = []
    processed = 0
    skipped = 0
    
    print(f"\n🔄 Processing samples (max: {max_samples})...")
    
    for idx, row in df.iterrows():
        if processed >= max_samples:
            break
        
        # Get audio file path
        audio_filename = row.get('path', '')
        if not audio_filename:
            skipped += 1
            continue
        
        audio_path = os.path.join(audio_dir, audio_filename)
        
        # Check if file exists
        if not os.path.exists(audio_path):
            skipped += 1
            continue
        
        # Get text
        text = str(row.get('sentence', '')).strip()
        if not text or len(text) < 10:  # Skip very short texts
            skipped += 1
            continue
        
        # Check duration
        try:
            wav, sr = torchaudio.load(audio_path)
            duration = wav.shape[-1] / sr
            if duration < min_duration or duration > max_duration:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue
        
        # Convert to WAV
        wav_filename = os.path.splitext(audio_filename)[0] + '.wav'
        wav_path = os.path.join(output_dir, wav_filename)
        
        # Skip if already converted
        if os.path.exists(wav_path):
            samples.append((wav_path, text))
            processed += 1
            if processed % 50 == 0:
                print(f"   Processed: {processed}/{max_samples}")
            continue
        
        # Convert MP3 to WAV
        if convert_mp3_to_wav(audio_path, wav_path):
            samples.append((wav_path, text))
            processed += 1
            if processed % 50 == 0:
                print(f"   Processed: {processed}/{max_samples}")
        else:
            skipped += 1
    
    print(f"\n✅ Processed: {processed} samples")
    print(f"   Skipped: {skipped} samples")
    
    return samples

def add_to_csv(samples: List[Tuple[str, str]], csv_path: str, relative_to: str = None):
    """Add samples to existing CSV or create new one."""
    if relative_to is None:
        relative_to = os.path.dirname(csv_path)
    
    # Read existing CSV if it exists
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\n📋 Existing CSV has {len(df)} samples")
    else:
        df = pd.DataFrame(columns=['path', 'text'])
        print(f"\n📋 Creating new CSV")
    
    # Add new samples
    new_rows = []
    for wav_path, text in samples:
        # Make path relative to CSV location
        if os.path.isabs(wav_path):
            rel_path = os.path.relpath(wav_path, relative_to)
        else:
            rel_path = wav_path
        
        new_rows.append({'path': rel_path, 'text': text})
    
    new_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    
    # Save
    combined_df.to_csv(csv_path, index=False)
    print(f"✅ Added {len(new_rows)} samples to {csv_path}")
    print(f"   Total samples: {len(combined_df)}")

def main():
    parser = argparse.ArgumentParser(description='Integrate Common Voice dataset')
    parser.add_argument('--tsv_path', type=str, required=True,
                       help='Path to Common Voice TSV file')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing Common Voice audio files')
    parser.add_argument('--output_dir', type=str, default='data/raw/commonvoice/',
                       help='Directory to save converted WAV files')
    parser.add_argument('--target_csv', type=str, default='data/manifests/train.csv',
                       help='CSV file to add samples to')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum number of samples to process')
    parser.add_argument('--min_duration', type=float, default=5.0,
                       help='Minimum audio duration in seconds')
    parser.add_argument('--max_duration', type=float, default=16.0,
                       help='Maximum audio duration in seconds')
    parser.add_argument('--instructions', action='store_true',
                       help='Show download instructions')
    
    args = parser.parse_args()
    
    if args.instructions:
        download_commonvoice_instructions()
        return
    
    # Process Common Voice
    samples = process_commonvoice(
        tsv_path=args.tsv_path,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
    
    if not samples:
        print("\n❌ No samples processed. Check paths and try again.")
        return
    
    # Add to CSV
    csv_dir = os.path.dirname(args.target_csv)
    add_to_csv(samples, args.target_csv, relative_to=csv_dir)
    
    print("\n" + "=" * 70)
    print("✅ COMMON VOICE INTEGRATION COMPLETE")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Check data quality: python check_data_quality.py")
    print(f"2. Train model: python run_all.py")
    print(f"\nNote: You're still training from scratch!")
    print("   Common Voice just provides more diverse data.")
    print("   This is still impressive for VTU evaluation.")

if __name__ == "__main__":
    main()

