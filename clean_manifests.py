import os
import pandas as pd
import torchaudio
import re

TRAIN_CSV = "data/manifests/train.csv"
VAL_CSV = "data/manifests/val.csv"

def fix_path_and_check_file(file_path, project_root):
    """Fix missing .wav extension and check if file exists and is valid."""
    original_path = file_path
    
    # Fix missing .wav extension
    if not file_path.endswith('.wav'):
        file_path = file_path + '.wav'
    
    # Resolve full path (paths in CSV are relative to project root)
    if not os.path.isabs(file_path):
        full_path = os.path.join(project_root, file_path)
    else:
        full_path = file_path
    
    # Normalize path separators
    full_path = os.path.normpath(full_path)
    
    # Check if file exists
    if not os.path.exists(full_path):
        return None, "File not found"
    
    # Try to load the file
    try:
        wav, sr = torchaudio.load(full_path)
        if wav.numel() == 0:
            return None, "Empty audio file"
        # If we can load it, return the fixed path (relative path)
        return file_path, None
    except Exception as e:
        return None, f"Corrupted: {str(e)[:50]}"

def clean_csv(csv_path, output_path=None):
    """Clean CSV by removing invalid entries and fixing paths."""
    if output_path is None:
        output_path = csv_path
    
    print(f"\n📋 Cleaning {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️  File not found: {csv_path}")
        return 0, 0
    
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    # Get project root (where script is run from)
    project_root = os.getcwd()
    
    valid_rows = []
    removed_count = 0
    
    for idx, row in df.iterrows():
        file_path = str(row["path"])
        text = str(row["text"]).strip()
        
        # Fix path and check validity
        fixed_path, error = fix_path_and_check_file(file_path, base_dir)
        
        if fixed_path and error is None:
            # Path is valid, add to valid rows
            valid_rows.append({"path": fixed_path, "text": text})
        else:
            removed_count += 1
            if removed_count <= 10:  # Print first 10 removals
                print(f"  ❌ Removing row {idx+2}: {file_path} ({error})")
    
    # Create new DataFrame with valid rows
    cleaned_df = pd.DataFrame(valid_rows)
    
    # Save cleaned CSV
    cleaned_df.to_csv(output_path, index=False)
    
    print(f"  ✅ Kept {len(cleaned_df)}/{original_count} entries")
    print(f"  ❌ Removed {removed_count} invalid entries")
    
    return len(cleaned_df), removed_count

def main():
    print("=" * 70)
    print("CLEANING MANIFEST FILES")
    print("=" * 70)
    
    train_valid, train_removed = clean_csv(TRAIN_CSV)
    val_valid, val_removed = clean_csv(VAL_CSV)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Train CSV: {train_valid} valid entries ({train_removed} removed)")
    print(f"Val CSV: {val_valid} valid entries ({val_removed} removed)")
    print(f"Total valid: {train_valid + val_valid} entries")
    
    if train_removed > 0 or val_removed > 0:
        print("\n💡 Next step: Run 'python check_data_quality.py' to verify")
    else:
        print("\n✅ All entries are valid!")

if __name__ == "__main__":
    main()

