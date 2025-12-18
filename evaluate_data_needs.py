import pandas as pd
import json

print("=" * 70)
print("FINAL EVALUATION: Sample Requirements for Superb Performance")
print("=" * 70)

train = pd.read_csv('data/manifests/train.csv')
val = pd.read_csv('data/manifests/val.csv')

current_train = len(train)
current_val = len(val)
current_total = current_train + current_val

avg_words = train['text'].str.split().str.len().mean()
total_words = train['text'].str.split().str.len().sum()
unique_words = set()
for text in train['text']:
    unique_words.update(str(text).lower().split())

print(f"\n📊 CURRENT DATASET ANALYSIS:")
print(f"  Train samples: {current_train}")
print(f"  Val samples: {current_val}")
print(f"  Total samples: {current_total}")
print(f"  Average words per sample: {avg_words:.1f}")
print(f"  Total words: {total_words}")
print(f"  Unique words: {len(unique_words)}")

print(f"\n🎯 TARGET: 'Superb Performance' Definition:")
print(f"  - WER < 0.3 (30% word error rate)")
print(f"  - CER < 0.2 (20% character error rate)")
print(f"  - Works for new speakers")
print(f"  - Handles unseen sentences")
print(f"  - Production-quality transcriptions")

print(f"\n📈 SAMPLE REQUIREMENTS ANALYSIS:")

print(f"\n1. MINIMUM VIABLE (Basic Functionality):")
min_samples = 500
print(f"   Samples needed: {min_samples}")
print(f"   Additional needed: {min_samples - current_total} samples")
print(f"   Expected WER: 0.5-0.7")
print(f"   Status: Model will learn, but accuracy limited")

print(f"\n2. GOOD PERFORMANCE (VTU Demo Quality):")
good_samples = 1000
print(f"   Samples needed: {good_samples}")
print(f"   Additional needed: {good_samples - current_total} samples")
print(f"   Expected WER: 0.3-0.5")
print(f"   Status: Good for demo, readable transcriptions")

print(f"\n3. EXCELLENT PERFORMANCE (Superb as Planned):")
excellent_samples = 2000
print(f"   Samples needed: {excellent_samples}")
print(f"   Additional needed: {excellent_samples - current_total} samples")
print(f"   Expected WER: 0.2-0.3")
print(f"   Status: Production-quality, works reliably")

print(f"\n4. INDUSTRY STANDARD (Research/Production):")
industry_samples = 5000
print(f"   Samples needed: {industry_samples}+")
print(f"   Additional needed: {industry_samples - current_total}+ samples")
print(f"   Expected WER: < 0.2")
print(f"   Status: Research/industry level")

print(f"\n💡 RECOMMENDATIONS:")

if current_total < 500:
    print(f"   ⚠️  Current dataset ({current_total}) is TOO SMALL for reliable learning")
    print(f"   ✅ Minimum: Add {500 - current_total} more samples (total: 500)")
    print(f"   ✅ Recommended: Add {1000 - current_total} more samples (total: 1000)")
    print(f"   ✅ Superb: Add {2000 - current_total} more samples (total: 2000)")
elif current_total < 1000:
    print(f"   ⚠️  Current dataset ({current_total}) is MINIMAL")
    print(f"   ✅ Good: Add {1000 - current_total} more samples (total: 1000)")
    print(f"   ✅ Superb: Add {2000 - current_total} more samples (total: 2000)")
else:
    print(f"   ✅ Current dataset ({current_total}) is reasonable")
    print(f"   ✅ For superb: Add {2000 - current_total} more samples (total: 2000)")

print(f"\n📝 FACTORS AFFECTING REQUIREMENTS:")
print(f"   - Architecture: CNN-LSTM-CTC (moderate complexity)")
print(f"   - Vocabulary: Small (33 chars) - HELPS reduce requirements")
print(f"   - Task: Character-level ASR - MODERATE requirements")
print(f"   - Domain: General speech - STANDARD requirements")
print(f"   - Speakers: Multi-speaker - INCREASES requirements")

print(f"\n🎓 FOR VTU PROJECT:")
print(f"   Minimum acceptable: 500-700 samples")
print(f"   Good demo quality: 1000-1500 samples")
print(f"   Superb as planned: 2000+ samples")
print(f"   Industry level: 5000+ samples")

print(f"\n⏱️  ESTIMATED EFFORT:")
additional_700 = 700 - current_total
additional_1000 = 1000 - current_total
additional_2000 = 2000 - current_total

print(f"   To reach 700 samples: {additional_700} more recordings")
print(f"     Time estimate: {additional_700 * 0.5:.0f}-{additional_700 * 1:.0f} minutes")
print(f"   To reach 1000 samples: {additional_1000} more recordings")
print(f"     Time estimate: {additional_1000 * 0.5:.0f}-{additional_1000 * 1:.0f} minutes")
print(f"   To reach 2000 samples: {additional_2000} more recordings")
print(f"     Time estimate: {additional_2000 * 0.5:.0f}-{additional_2000 * 1:.0f} minutes")

print(f"\n" + "=" * 70)
print("FINAL VERDICT:")
print("=" * 70)
print(f"For 'SUPERB as planned' performance:")
print(f"  🎯 TARGET: 2000 total samples")
print(f"  📊 CURRENT: {current_total} samples")
print(f"  ➕ NEEDED: {2000 - current_total} additional samples")
print(f"  ⏱️  EFFORT: ~{(2000 - current_total) * 0.75:.0f} minutes of recording")
print(f"\nFor 'GOOD demo' performance:")
print(f"  🎯 TARGET: 1000 total samples")
print(f"  📊 CURRENT: {current_total} samples")
print(f"  ➕ NEEDED: {1000 - current_total} additional samples")
print(f"  ⏱️  EFFORT: ~{(1000 - current_total) * 0.75:.0f} minutes of recording")

