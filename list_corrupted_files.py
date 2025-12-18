import os
import pandas as pd
import torch
import torchaudio

TRAIN_CSV = "data/manifests/train.csv"
VAL_CSV = "data/manifests/val.csv"

corrupted_files = []
missing_files = []

def check_file(file_path, row_num, csv_name):
    if not os.path.exists(file_path):
        missing_files.append((file_path, row_num, csv_name, "File not found"))
        return
    
    try:
        wav, sr = torchaudio.load(file_path)
        if wav.numel() == 0:
            corrupted_files.append((file_path, row_num, csv_name, "Empty audio file"))
    except Exception as e:
        error_msg = str(e)
        if "Format not recognised" in error_msg:
            corrupted_files.append((file_path, row_num, csv_name, "Format not recognized"))
        else:
            corrupted_files.append((file_path, row_num, csv_name, f"Error: {error_msg[:50]}"))

print("=" * 70)
print("CORRUPTED FILES REPORT")
print("=" * 70)

for csv_path, csv_name in [(TRAIN_CSV, "train.csv"), (VAL_CSV, "val.csv")]:
    if not os.path.exists(csv_path):
        print(f"\n⚠️  CSV not found: {csv_path}")
        continue
    
    df = pd.read_csv(csv_path)
    print(f"\n📋 Checking {csv_name} ({len(df)} files)...")
    
    for idx, row in df.iterrows():
        file_path = str(row["path"])
        
        if not os.path.isabs(file_path):
            full_path = os.path.join(os.getcwd(), file_path)
        else:
            full_path = file_path
        
        check_file(full_path, idx + 2, csv_name)

print("\n" + "=" * 70)
print("CORRUPTED FILES (Cannot be loaded)")
print("=" * 70)

if corrupted_files:
    print(f"\nTotal corrupted files: {len(corrupted_files)}\n")
    
    for i, (file_path, row_num, csv_name, reason) in enumerate(corrupted_files, 1):
        print(f"{i:3d}. {file_path}")
        print(f"     CSV: {csv_name}, Row: {row_num}, Reason: {reason}")
    
    print(f"\n📝 List for removal:")
    print("=" * 70)
    for file_path, _, _, _ in corrupted_files:
        print(file_path)
else:
    print("\n✅ No corrupted files found!")

print("\n" + "=" * 70)
print("MISSING FILES (Not found on disk)")
print("=" * 70)

if missing_files:
    print(f"\nTotal missing files: {len(missing_files)}\n")
    
    for i, (file_path, row_num, csv_name, reason) in enumerate(missing_files, 1):
        print(f"{i:3d}. {file_path}")
        print(f"     CSV: {csv_name}, Row: {row_num}")
    
    print(f"\n📝 List for removal:")
    print("=" * 70)
    for file_path, _, _, _ in missing_files:
        print(file_path)
else:
    print("\n✅ No missing files found!")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Corrupted files: {len(corrupted_files)}")
print(f"Missing files: {len(missing_files)}")
print(f"Total problematic: {len(corrupted_files) + len(missing_files)}")

if corrupted_files or missing_files:
    print("\n💡 Next steps:")
    print("   1. Review the corrupted/missing files above")
    print("   2. Either fix/re-record these files")
    print("   3. Or remove them from train.csv/val.csv")
    print("   4. Run 'python check_data_quality.py' again to verify")

