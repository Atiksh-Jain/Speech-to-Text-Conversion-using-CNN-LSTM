"""
One-command script to prepare everything for tomorrow.
Sets up infrastructure and generates TTS samples now.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Running: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("=" * 70)
    print("PREPARING INFRASTRUCTURE FOR TOMORROW")
    print("=" * 70)
    print("\nThis will:")
    print("1. Set up data directories")
    print("2. Generate 200-300 TTS samples")
    print("3. Prepare everything for your 800 recordings tomorrow")
    print("\nPress ENTER to continue or Ctrl+C to cancel...")
    input()
    
    # Step 1: Setup directories
    if not run_command(
        "python scripts/setup_data_infrastructure.py",
        "Step 1: Setting up data directories"
    ):
        print("\n❌ Failed to set up directories. Check errors above.")
        return
    
    # Step 2: Generate TTS samples
    print("\n" + "="*70)
    print("Step 2: Generating TTS samples")
    print("="*70)
    print("\nChoose TTS method:")
    print("1. pyttsx3 (offline, no internet needed) - Recommended")
    print("2. gTTS (online, requires internet)")
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    
    method = "pyttsx3" if choice == "1" else "gtts"
    num_samples = input("Number of TTS samples (default=200): ").strip() or "200"
    
    cmd = f"python scripts/generate_tts_samples.py --num_samples {num_samples} --method {method}"
    
    if not run_command(cmd, f"Generating {num_samples} TTS samples using {method}"):
        print("\n⚠️  TTS generation failed. You can:")
        print("   - Install missing packages: pip install pyttsx3 gtts")
        print("   - Or skip TTS and just use your 800 recordings")
        print("   - Or generate TTS later")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ PREPARATION COMPLETE")
    print("=" * 70)
    print("\n📋 What's ready:")
    print("   ✅ Data directories created")
    print("   ✅ TTS samples generated (if successful)")
    print("   ✅ Infrastructure ready")
    print("\n📋 What to do tomorrow:")
    print("   1. Record 800 samples")
    print("   2. Place them in: data/raw/user_recorded/")
    print("   3. Update: data/manifests/train.csv with paths")
    print("   4. Run: python check_data_quality.py")
    print("   5. Run: python run_all.py")
    print("\n💡 Tips:")
    print("   - Use: python scripts/record_with_prompts.py (for structured recording)")
    print("   - Or manually add files and update CSV")
    print("   - CSV format: path,text")
    print("   - Example: raw/user_recorded/sample_001.wav,hello world")
    print("\n🚀 You're all set! Add your recordings tomorrow and train!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)

