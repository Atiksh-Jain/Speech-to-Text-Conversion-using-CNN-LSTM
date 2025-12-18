import json
import os

if os.path.exists("training_history.json"):
    with open("training_history.json", "r") as f:
        history = json.load(f)
    
    epochs = history.get("epochs", [])
    if len(epochs) < 2:
        print("Not enough data")
        exit(0)
    
    # Get recent epochs with WER data
    recent = [e for e in epochs if 'val_wer' in e and e['val_wer'] < 1.5][-10:]
    
    if len(recent) < 2:
        print("Not enough recent epochs with valid WER")
        exit(0)
    
    print("=" * 70)
    print("WER PROJECTION ANALYSIS")
    print("=" * 70)
    
    # Calculate improvement rate
    wer_values = [e['val_wer'] for e in recent]
    epochs_nums = [e['epoch'] for e in recent]
    
    # Linear trend
    if len(wer_values) >= 2:
        first_wer = wer_values[0]
        last_wer = wer_values[-1]
        first_epoch = epochs_nums[0]
        last_epoch = epochs_nums[-1]
        
        epochs_diff = last_epoch - first_epoch
        wer_improvement = first_wer - last_wer
        improvement_per_epoch = wer_improvement / epochs_diff if epochs_diff > 0 else 0
        
        print(f"\nCurrent Status:")
        print(f"  Epoch {first_epoch}: WER = {first_wer:.4f}")
        print(f"  Epoch {last_epoch}: WER = {last_wer:.4f}")
        print(f"  Improvement: {wer_improvement:.4f} over {epochs_diff} epochs")
        print(f"  Rate: {improvement_per_epoch:.4f} WER per epoch")
        
        # Project to epoch 30
        current_epoch = last_epoch
        current_wer = last_wer
        epochs_remaining = 30 - current_epoch
        
        if improvement_per_epoch > 0:
            projected_wer_30 = max(0, current_wer - (improvement_per_epoch * epochs_remaining))
        else:
            projected_wer_30 = current_wer
        
        print(f"\n{'=' * 70}")
        print(f"PROJECTION TO EPOCH 30:")
        print(f"{'=' * 70}")
        print(f"  Current: Epoch {current_epoch}, WER = {current_wer:.4f}")
        print(f"  Remaining: {epochs_remaining} epochs")
        print(f"  Projected WER at epoch 30: {projected_wer_30:.4f}")
        print(f"  Target: < 0.35")
        
        if projected_wer_30 < 0.35:
            print(f"\n✅ OPTIMISTIC: At current rate, we should reach target!")
        elif projected_wer_30 < 0.50:
            print(f"\n⚠️  REALISTIC: May reach 0.35-0.50 range (still good!)")
        else:
            print(f"\n⚠️  CAUTIOUS: May not reach 0.35, but will improve significantly")
        
        print(f"\n{'=' * 70}")
        print("STRATEGIES IF TARGET NOT REACHED:")
        print(f"{'=' * 70}")
        
        print("\n1. CONTINUE TRAINING BEYOND 30 EPOCHS:")
        print("   - Training can continue until early stopping triggers")
        print("   - Early stopping patience: 15 epochs")
        print("   - Command: Just run training again (it will continue from checkpoint)")
        
        print("\n2. ACCEPTABLE WER LEVELS:")
        print("   - WER < 0.20: Excellent (commercial quality)")
        print("   - WER 0.20-0.35: Very good (research/academic)")
        print("   - WER 0.35-0.50: Good (demonstration/proof-of-concept)")
        print("   - WER 0.50-0.70: Acceptable (early stage research)")
        print("   - WER > 0.70: Needs improvement")
        
        print("\n3. IF WER STUCK ABOVE 0.50:")
        print("   a) Add more data (Common Voice integration)")
        print("   b) Adjust learning rate (try lower: 5e-4)")
        print("   c) Increase model capacity")
        print("   d) Try different augmentation strategies")
        
        print("\n4. FOR VTU PROJECT:")
        print("   - WER 0.35-0.50 is still EXCELLENT for a final year project")
        print("   - Focus on:")
        print("     * Clear methodology documentation")
        print("     * Comparison with baseline")
        print("     * Analysis of errors (what words/phrases are hard)")
        print("     * Discussion of limitations and future work")
        
        print("\n5. REALISTIC EXPECTATION:")
        print(f"   - Current trajectory suggests WER ~{projected_wer_30:.2f} at epoch 30")
        print("   - This is GOOD progress for a CNN-LSTM model")
        print("   - Many research papers report WER 0.30-0.60 on similar setups")
        
        print(f"\n{'=' * 70}")
        print("RECOMMENDATION:")
        print(f"{'=' * 70}")
        print("✅ Continue training to 30 epochs")
        print("✅ If WER > 0.35 but < 0.50, that's still a SUCCESS")
        print("✅ Document the results and analyze what works/doesn't work")
        print("✅ Consider continuing to 40-50 epochs if still improving")
        print("✅ For VTU project, focus on methodology and analysis, not just final WER")

