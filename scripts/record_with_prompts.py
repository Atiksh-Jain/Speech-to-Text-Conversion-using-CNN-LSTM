"""
Enhanced recording script with structured prompts.
Similar to Mimic Recording Studio approach.
"""

import os
import sounddevice as sd
import soundfile as sf
import argparse
from datetime import datetime
import pandas as pd

# Structured prompts for recording
PROMPTS = [
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
]

def record_audio(duration: float = 5.0, sample_rate: int = 16000):
    """Record audio from microphone."""
    print(f"\n🎤 Recording for {duration} seconds...")
    print("   Speak now!")
    
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    
    print("   ✅ Recording complete")
    return audio.flatten()

def save_recording(audio, output_path: str, sample_rate: int = 16000):
    """Save recorded audio to file."""
    sf.write(output_path, audio, sample_rate)
    print(f"   💾 Saved: {output_path}")

def record_with_prompts(
    output_dir: str,
    csv_path: str,
    num_samples: int = 10,
    duration: float = 5.0,
    sample_rate: int = 16000
):
    """Record audio samples with structured prompts."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("STRUCTURED RECORDING WITH PROMPTS")
    print("=" * 70)
    print(f"\n📋 Configuration:")
    print(f"   Output directory: {output_dir}")
    print(f"   CSV file: {csv_path}")
    print(f"   Number of samples: {num_samples}")
    print(f"   Duration per sample: {duration} seconds")
    print(f"   Sample rate: {sample_rate} Hz")
    
    # Load or create CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        start_idx = len(df) + 1
        print(f"\n📋 Existing CSV has {len(df)} samples")
    else:
        df = pd.DataFrame(columns=['path', 'text'])
        start_idx = 1
        print(f"\n📋 Creating new CSV")
    
    print(f"\n🎤 Starting recording session...")
    print(f"   You'll record {num_samples} samples")
    print(f"   Each sample is {duration} seconds")
    print(f"   Read the prompts clearly\n")
    
    new_samples = []
    
    for i in range(num_samples):
        prompt_idx = (start_idx + i - 1) % len(PROMPTS)
        prompt = PROMPTS[prompt_idx]
        
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*70}")
        print(f"\n📝 Prompt to read:")
        print(f"   \"{prompt}\"")
        print(f"\n   Press ENTER when ready to record...")
        input()
        
        # Record
        audio = record_audio(duration, sample_rate)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_{start_idx + i:04d}_{timestamp}.wav"
        filepath = os.path.join(output_dir, filename)
        save_recording(audio, filepath, sample_rate)
        
        # Add to CSV
        rel_path = os.path.relpath(filepath, os.path.dirname(csv_path))
        new_samples.append({'path': rel_path, 'text': prompt})
        
        print(f"   ✅ Sample {i+1} recorded and saved")
    
    # Update CSV
    new_df = pd.DataFrame(new_samples)
    combined_df = pd.concat([df, new_df], ignore_index=True)
    combined_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("✅ RECORDING SESSION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"   Recorded: {num_samples} samples")
    print(f"   Total in CSV: {len(combined_df)} samples")
    print(f"   CSV saved: {csv_path}")
    print(f"\nNext steps:")
    print(f"1. Check data quality: python check_data_quality.py")
    print(f"2. Record more samples if needed")
    print(f"3. Train model: python run_all.py")

def main():
    parser = argparse.ArgumentParser(description='Record audio with structured prompts')
    parser.add_argument('--output_dir', type=str, default='data/raw/user_recorded/',
                       help='Directory to save recordings')
    parser.add_argument('--csv_path', type=str, default='data/manifests/train.csv',
                       help='CSV file to update')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to record')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration of each recording in seconds')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Sample rate for recording')
    
    args = parser.parse_args()
    
    try:
        record_with_prompts(
            output_dir=args.output_dir,
            csv_path=args.csv_path,
            num_samples=args.num_samples,
            duration=args.duration,
            sample_rate=args.sample_rate
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Recording interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

