import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder import RotaryPositionalEmbedding, RelativePositionBias, MultiHeadAttention, PositionwiseFeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, use_relative_position=False, max_length=128, use_rope=False):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, num_heads, dropout, use_relative_position, max_length, use_rope)
        self.cross_attention = MultiHeadAttention(
            d_model, num_heads, dropout, use_relative_position, max_length, use_rope=False)  # Cross-attention doesn't use RoPE
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Masked self-attention with residual connection and layer norm
        self_attn_output, _ = self.self_attention(x, x, x, self_attention_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention with residual connection and layer norm
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, cross_attention_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, cross_attn_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_length=5000, dropout=0.1, positional_encoding='rope'):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = positional_encoding
        if positional_encoding == 'rope':
            self.pos_encoding = None  # Don't add to embeddings
            use_relative_position = False
            use_rope = True
        else:  # relative position bias
            self.pos_encoding = None
            use_relative_position = True
            use_rope = False
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, use_relative_position, max_length, use_rope)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt, encoder_output, self_attention_mask=None, cross_attention_mask=None):
        # Embedding and positional encoding
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Don't add positional encoding for RoPE - it's applied in attention
        # Don't add anything for relative position either - it's applied in attention
        
        x = self.dropout(x)
        
        # Pass through decoder layers
        cross_attention_weights = []
        for layer in self.layers:
            x, cross_attn_weights = layer(x, encoder_output, self_attention_mask, cross_attention_mask)
            cross_attention_weights.append(cross_attn_weights)
        
        return x, cross_attention_weights

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_length=5000, dropout=0.1, positional_encoding='rope'):
        super().__init__()
        
        from encoder import Encoder
        
        self.encoder = Encoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, d_ff,
            max_length, dropout, positional_encoding
        )
        
        self.decoder = Decoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers, d_ff,
            max_length, dropout, positional_encoding
        )
        
        self.projection = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.init_parameters()
        
    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_masks(self, src, tgt):
        # Source padding mask
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Target padding mask
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        # Target look-ahead mask
        seq_len = tgt.size(1)
        look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0).to(tgt.device)
        
        # Combined target mask
        tgt_mask = tgt_mask & ~look_ahead_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # Encoder
        encoder_output = self.encoder(src, src_mask)
        
        # Decoder
        decoder_output, cross_attention_weights = self.decoder(
            tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.projection(decoder_output)
        
        return output, cross_attention_weights
    
    def encode(self, src):
        """Encode source sequence"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return self.encoder(src, src_mask)
    
    def decode_step(self, tgt, encoder_output, src_mask):
        """Single decoding step"""
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        
        # Look-ahead mask
        look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0).to(tgt.device)
        tgt_mask = tgt_mask & ~look_ahead_mask
        
        # Decoder
        decoder_output, _ = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.projection(decoder_output)
        
        return output

class DecodingStrategy:
    def __init__(self, model, src_tokenizer, tgt_tokenizer, device, max_length=100):
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        self.max_length = max_length
        
    def greedy_decode(self, src_sentence):
        """Greedy decoding strategy"""
        self.model.eval()
        
        # Tokenize source
        src_indices = self.src_tokenizer.encode(src_sentence, add_special_tokens=False)
        src = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # Encode source
        encoder_output = self.model.encode(src)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Initialize target with SOS token
        tgt = torch.tensor([[self.tgt_tokenizer.word2idx['<sos>']]], dtype=torch.long).to(self.device)
        
        for _ in range(self.max_length - 1):
            # Get next token probabilities
            output = self.model.decode_step(tgt, encoder_output, src_mask)
            
            # Select token with highest probability
            next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.tgt_tokenizer.word2idx['<eos>']:
                break
        
        # Decode to string
        output_indices = tgt[0].cpu().numpy().tolist()
        return self.tgt_tokenizer.decode(output_indices)
    
    def beam_search_decode(self, src_sentence, beam_size=5):
        """Improved beam search decoding strategy with proper batching and score normalization"""
        self.model.eval()
        
        # Tokenize source
        src_indices = self.src_tokenizer.encode(src_sentence, add_special_tokens=False)
        src = torch.tensor([src_indices], dtype=torch.long).to(self.device)
        
        # Encode source
        encoder_output = self.model.encode(src)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Initialize beam search
        sos_token = self.tgt_tokenizer.word2idx['<sos>']
        eos_token = self.tgt_tokenizer.word2idx['<eos>']
        pad_token = self.tgt_tokenizer.word2idx['<pad>']
        
        vocab_size = len(self.tgt_tokenizer.word2idx)
        
        # Start with SOS token
        sequences = torch.full((beam_size, 1), sos_token, dtype=torch.long, device=self.device)
        scores = torch.zeros(beam_size, device=self.device)
        scores[1:] = -float('inf')  # Only first beam is active initially
        
        completed_sequences = []
        
        for step in range(self.max_length - 1):
            if len(completed_sequences) >= beam_size:
                break
                
            # Get current sequence length
            seq_len = sequences.size(1)
            
            # Expand encoder output for all beams
            expanded_encoder_output = encoder_output.expand(beam_size, -1, -1)
            expanded_src_mask = src_mask.expand(beam_size, -1, -1, -1)
            
            # Decode current sequences
            outputs = self.model.decode_step(sequences, expanded_encoder_output, expanded_src_mask)
            
            # Get log probabilities for last token
            log_probs = F.log_softmax(outputs[:, -1, :], dim=-1)  # [beam_size, vocab_size]
            
            # Add current scores to log probabilities
            candidate_scores = scores.unsqueeze(1) + log_probs  # [beam_size, vocab_size]
            
            # Flatten to get all candidates
            candidate_scores = candidate_scores.view(-1)  # [beam_size * vocab_size]
            
            # Get top beam_size candidates
            top_scores, top_indices = torch.topk(candidate_scores, beam_size)
            
            # Convert back to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size
            
            # Create new sequences
            new_sequences = []
            new_scores = []
            
            for i, (beam_idx, token_idx, score) in enumerate(zip(beam_indices, token_indices, top_scores)):
                # Get previous sequence
                prev_seq = sequences[beam_idx]
                
                # Create new sequence
                new_seq = torch.cat([prev_seq, token_idx.unsqueeze(0)])
                
                # Check if sequence is completed
                if token_idx == eos_token:
                    # Apply length normalization to completed sequences
                    normalized_score = score / len(new_seq)
                    completed_sequences.append((new_seq, normalized_score.item()))
                else:
                    new_sequences.append(new_seq)
                    new_scores.append(score)
            
            # If we don't have enough active sequences, break
            if not new_sequences:
                break
                
            # Pad sequences to same length
            max_len = max(len(seq) for seq in new_sequences)
            padded_sequences = []
            
            for seq in new_sequences:
                padded_seq = torch.cat([seq, torch.full((max_len - len(seq),), pad_token, dtype=torch.long, device=self.device)])
                padded_sequences.append(padded_seq)
            
            # Keep only top beam_size sequences
            if len(padded_sequences) > beam_size:
                # Sort by score and keep top beam_size
                scored_seqs = list(zip(padded_sequences, new_scores))
                scored_seqs.sort(key=lambda x: x[1], reverse=True)
                padded_sequences = [seq for seq, _ in scored_seqs[:beam_size]]
                new_scores = [score for _, score in scored_seqs[:beam_size]]
            
            sequences = torch.stack(padded_sequences[:beam_size])
            scores = torch.tensor(new_scores[:beam_size], device=self.device)
            
            # Pad with dummy sequences if needed
            while len(scores) < beam_size:
                sequences = torch.cat([sequences, torch.full((1, sequences.size(1)), pad_token, dtype=torch.long, device=self.device)])
                scores = torch.cat([scores, torch.tensor([-float('inf')], device=self.device)])
        
        # Add remaining sequences to completed
        for i, (seq, score) in enumerate(zip(sequences, scores)):
            if score > -float('inf'):
                # Remove padding
                seq_no_pad = []
                for token in seq:
                    if token == pad_token:
                        break
                    seq_no_pad.append(token.item())
                
                if seq_no_pad:
                    normalized_score = score / len(seq_no_pad)
                    completed_sequences.append((torch.tensor(seq_no_pad, device=self.device), normalized_score.item()))
        
        # Return best sequence
        if completed_sequences:
            best_seq, _ = max(completed_sequences, key=lambda x: x[1])
            output_indices = best_seq.cpu().numpy().tolist()
        else:
            # Fallback to first sequence if nothing completed
            output_indices = [sos_token, eos_token]
        
        return self.tgt_tokenizer.decode(output_indices)
    
    