import torch
import torch.nn.functional as F
from utils import Tokenizer
from decoder import Transformer, DecodingStrategy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizers from the RoPE model directory
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()
src_tokenizer.load('models/rope_final/src_tokenizer.pkl')
tgt_tokenizer.load('models/rope_final/tgt_tokenizer.pkl')

# Create RoPE model (not relative)
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
    positional_encoding='rope'  # Make sure this matches
).to(device)

# Load weights
checkpoint = torch.load('models/rope_final/best_model.pt', map_location=device)
state_dict = checkpoint['model_state_dict']
if any(key.startswith('module.') for key in state_dict.keys()):
    state_dict = {key[7:]: value for key, value in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Create DecodingStrategy object for beam search
decoder = DecodingStrategy(model, src_tokenizer, tgt_tokenizer, device, max_length=100)

# Test sentences
test_sentences = [

    "kissa",      # cat
    "koira",      # dog  
    "auto",       # car
    "talo",       # house
    "vesi"        # water

]

print("=== TESTING ROPE MODEL WITH BEAM SEARCH ===")
for i, sentence in enumerate(test_sentences):
    print(f"\n--- Test {i+1}: '{sentence}' ---")
    
    # Greedy decoding
    greedy_translation = decoder.greedy_decode(sentence)
    print(f"Greedy:     '{greedy_translation}'")
    
    # Beam search decoding
    beam_translation = decoder.beam_search_decode(sentence, beam_size=5)
    print(f"Beam (5):   '{beam_translation}'")
    
    # Top-k sampling
    topk_translation = decoder.top_k_sampling_decode(sentence, k=50, temperature=1.0)
    print(f"Top-k (50): '{topk_translation}'")
    
    print("-" * 60)