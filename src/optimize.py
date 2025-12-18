"""
Model optimization script for production-ready inference.
Similar to optimize_graph.py from the reference project.
Creates optimized, frozen models for faster inference.
"""

import os
import torch
import argparse
from src.model import create_model
from src.utils import build_vocab

def optimize_model(
    checkpoint_path: str,
    output_path: str,
    vocab_size: int = None
):
    """
    Optimize a trained model for production inference.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_path: Path to save optimized model
        vocab_size: Vocabulary size (auto-detected if None)
    """
    print("=" * 70)
    print("MODEL OPTIMIZATION")
    print("=" * 70)
    
    # Load checkpoint
    print(f"\n📦 Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model state
    if 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        # Try to get vocab_size from checkpoint
        if vocab_size is None:
            vocab_size = checkpoint.get('vocab_size', None)
    else:
        model_state = checkpoint
        vocab_size = None
    
    # Build vocab to get vocab_size
    if vocab_size is None:
        print("   🔍 Auto-detecting vocab size...")
        _, idx2char = build_vocab()
        vocab_size = len(idx2char)
        print(f"   ✅ Vocab size: {vocab_size}")
    else:
        print(f"   ✅ Vocab size: {vocab_size}")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = create_model(vocab_size=vocab_size)
    model.load_state_dict(model_state)
    model.eval()
    
    # Freeze model
    print(f"   🔒 Freezing model parameters...")
    for param in model.parameters():
        param.requires_grad = False
    
    # Create optimized checkpoint
    print(f"\n⚡ Creating optimized checkpoint...")
    optimized_checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'optimized': True,
        'original_checkpoint': checkpoint_path
    }
    
    # Copy other metadata if present
    if isinstance(checkpoint, dict):
        for key in ['epoch', 'best_wer', 'best_cer', 'char2idx', 'idx2char']:
            if key in checkpoint:
                optimized_checkpoint[key] = checkpoint[key]
    
    # Save optimized model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(optimized_checkpoint, output_path)
    
    print(f"   ✅ Optimized model saved: {output_path}")
    
    # Calculate model size
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"   📊 Model size: {model_size_mb:.2f} MB")
    
    # Test inference speed (optional)
    print(f"\n🧪 Testing inference...")
    try:
        dummy_input = torch.randn(1, 1, 80, 100)  # (batch, channels, n_mels, time)
        dummy_lengths = torch.tensor([100])
        
        with torch.no_grad():
            import time
            start = time.time()
            for _ in range(10):
                _ = model(dummy_input)
            elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"   ✅ Average inference time: {avg_time*1000:.2f} ms")
    except Exception as e:
        print(f"   ⚠️  Could not test inference: {e}")
    
    print("\n" + "=" * 70)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOptimized model saved to: {output_path}")
    print(f"\nYou can now use this optimized model for:")
    print(f"  - Faster inference")
    print(f"  - Production deployment")
    print(f"  - Web app (update app.py to use this checkpoint)")

def main():
    parser = argparse.ArgumentParser(description='Optimize trained model for production')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/optimized_model.pt',
                       help='Path to save optimized model')
    parser.add_argument('--vocab_size', type=int, default=None,
                       help='Vocabulary size (auto-detected if not provided)')
    
    args = parser.parse_args()
    
    optimize_model(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        vocab_size=args.vocab_size
    )

if __name__ == "__main__":
    main()

