import torch
import torch.nn.functional as F
from utils import Tokenizer
from decoder import Transformer, DecodingStrategy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizers from relative model directory
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()
src_tokenizer.load('./models/relative_model/src_tokenizer.pkl')
tgt_tokenizer.load('./models/relative_model/tgt_tokenizer.pkl')

# Load relative position model
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
    positional_encoding='relative'  # Changed to relative
).to(device)

# Load weights from relative model
checkpoint = torch.load('./models/relative_model/best_model.pt', map_location=device)
state_dict = checkpoint['model_state_dict']
if any(key.startswith('module.') for key in state_dict.keys()):
    state_dict = {key[7:]: value for key, value in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# Test sentences
test_sentences = [
    "Auttaa työntekijöitä ja yrityksiä sopeutumaan talouden muutoksiin",
    "Suomi",
    "Euroopan sosiaalirahasto", 
    "Komissio kiinnittää huomiota",
    "Yritykset sopeutumaan muutoksiin"
]

print("=== TESTING RELATIVE POSITION MODEL ===")
for i, sentence in enumerate(test_sentences):
    print(f"\n--- Test {i+1}: '{sentence}' ---")
    
    # Encoding/generation loop
    src_indices = src_tokenizer.encode(sentence, add_special_tokens=False)
    print(f"Source encoded: {src_indices}")
    
    src = torch.tensor([src_indices], dtype=torch.long).to(device)
    encoder_output = model.encode(src)
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    
    sos_idx = tgt_tokenizer.word2idx['<sos>']
    eos_idx = tgt_tokenizer.word2idx['<eos>']
    tgt = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
    
    generated = []
    for step in range(10):  # Just 10 steps per sentence
        with torch.no_grad():
            output = model.decode_step(tgt, encoder_output, src_mask)
            logits = output[:, -1, :]
            
            next_token = logits.argmax(dim=-1, keepdim=True)
            next_token_id = next_token.item()
            
            tgt = torch.cat([tgt, next_token], dim=1)
            generated.append(next_token_id)
            
            if next_token_id == eos_idx:
                break
    
    final_translation = tgt_tokenizer.decode(tgt[0].cpu().tolist())
    print(f"Translation: '{final_translation}'")