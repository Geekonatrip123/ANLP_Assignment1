import torch
import torch.nn.functional as F
from utils import Tokenizer
from decoder import Transformer, DecodingStrategy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizers
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()
src_tokenizer.load('./models/rope_model/src_tokenizer.pkl')
tgt_tokenizer.load('./models/rope_model/tgt_tokenizer.pkl')

# Load model
model = Transformer(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    d_model=512,
    num_heads=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    d_ff=2048,
    max_length=128,
    dropout=0.1,
    positional_encoding='rope'
).to(device)

# Load weights
checkpoint = torch.load('./models/rope_model/best_model.pt', map_location=device)
state_dict = checkpoint['model_state_dict']
if any(key.startswith('module.') for key in state_dict.keys()):
    state_dict = {key[7:]: value for key, value in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Test with a sentence from your data
test_sentence = "Tämä ehdotus on väärä."  # This should be in your Finnish data

print(f"=== TESTING SENTENCE: '{test_sentence}' ===")

# Encode source
src_indices = src_tokenizer.encode(test_sentence, add_special_tokens=False)
print(f"Source encoded: {src_indices}")
print(f"Source decoded back: '{src_tokenizer.decode(src_indices)}'")

src = torch.tensor([src_indices], dtype=torch.long).to(device)
encoder_output = model.encode(src)
src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

# Initialize with SOS
sos_idx = tgt_tokenizer.word2idx['<sos>']
eos_idx = tgt_tokenizer.word2idx['<eos>']
tgt = torch.tensor([[sos_idx]], dtype=torch.long).to(device)

print(f"\nStarting generation (SOS={sos_idx}, EOS={eos_idx})...")

# Generate 5 tokens to see what happens
generated = []
for step in range(5):
    with torch.no_grad():
        output = model.decode_step(tgt, encoder_output, src_mask)
        logits = output[:, -1, :]  # Last token logits
        
        # Get top 5 predictions
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(5)
        
        print(f"\nStep {step}:")
        print("Top 5 predictions:")
        for i in range(5):
            token_id = top_indices[0, i].item()
            token_prob = top_probs[0, i].item()
            token_word = tgt_tokenizer.idx2word.get(token_id, f"UNK_{token_id}")
            print(f"  {token_word} (id={token_id}): {token_prob:.4f}")
        
        # Take the most likely token
        next_token = top_indices[0, 0].unsqueeze(0).unsqueeze(0)
        next_token_id = next_token.item()
        
        tgt = torch.cat([tgt, next_token], dim=1)
        generated.append(next_token_id)
        
        if next_token_id == eos_idx:
            print(f"EOS generated at step {step}")
            break

print(f"\nGenerated sequence: {generated}")
final_translation = tgt_tokenizer.decode(tgt[0].cpu().tolist())
print(f"Final translation: '{final_translation}'")