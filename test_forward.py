# test_forward.py
import torch
from decoder import Transformer
from utils import Tokenizer

# Create a tiny model for testing
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=128,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=256,
    positional_encoding='rope'
)

# Test forward pass
src = torch.randint(1, 100, (2, 10))  # batch=2, seq=10
tgt = torch.randint(1, 100, (2, 8))   # batch=2, seq=8

try:
    output, _ = model(src, tgt)
    print(f"Forward pass successful: {output.shape}")
    print(f"Output range: {output.min():.3f} to {output.max():.3f}")
    
    # Check if outputs are reasonable
    if torch.isnan(output).any():
        print("❌ NaN values in output")
    elif output.std() < 0.1:
        print("❌ Output has very low variance - possible dead neurons")
    else:
        print("✅ Output looks reasonable")
        
except Exception as e:
    print(f"❌ Forward pass failed: {e}")