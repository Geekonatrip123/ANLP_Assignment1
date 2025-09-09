import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle
import os
from collections import Counter, defaultdict
import re

class BPETokenizer:
    """
    Simple Byte Pair Encoding (BPE) implementation for subword tokenization
    """
    def __init__(self, vocab_size=32000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.idx2word = {}
        self.merges = []
        self.special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        
    def get_word_tokens(self, word):
        """Split word into characters with end-of-word marker"""
        return list(word[:-1]) + [word[-1] + '</w>']
    
    def get_pairs(self, word_tokens):
        """Get all adjacent pairs in word tokens"""
        pairs = set()
        prev_char = word_tokens[0]
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def merge_tokens(self, word_tokens, pair):
        """Apply a merge operation to word tokens"""
        new_tokens = []
        i = 0
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
                new_tokens.append(word_tokens[i] + word_tokens[i + 1])
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        return new_tokens
    
    def build_vocab(self, sentences, min_freq=2):
        """Train BPE on sentences"""
        print("Training BPE tokenizer...")
        
        # Initialize word frequencies
        word_freqs = defaultdict(int)
        for sentence in sentences:
            if isinstance(sentence, str) and sentence.strip():
                # Basic preprocessing
                sentence = sentence.strip().lower()
                words = sentence.split()
                for word in words:
                    word_freqs[word] += 1
        
        # Filter by minimum frequency
        word_freqs = {word: freq for word, freq in word_freqs.items() if freq >= min_freq}
        
        # Split words into characters
        vocab = defaultdict(int)
        for word, freq in word_freqs.items():
            word_tokens = self.get_word_tokens(word)
            for token in word_tokens:
                vocab[token] += freq
        
        # Get all pairs
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            word_tokens = self.get_word_tokens(word)
            word_pairs = self.get_pairs(word_tokens)
            for pair in word_pairs:
                pairs[pair] += freq
        
        # Merge most frequent pairs
        num_merges = self.vocab_size - len(self.special_tokens) - len(vocab)
        
        for i in range(num_merges):
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_freq:
                break
                
            # Merge the pair
            self.merges.append(best_pair)
            
            # Update word frequencies with merged token
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                word_tokens = self.get_word_tokens(word)
                # Apply all previous merges
                for merge in self.merges:
                    word_tokens = self.merge_tokens(word_tokens, merge)
                new_word_freqs[''.join(word_tokens)] = freq
            
            word_freqs = new_word_freqs
            
            # Update pairs
            pairs = defaultdict(int)
            for word, freq in word_freqs.items():
                word_tokens = list(word[:-1]) + [word[-1] + '</w>']
                # Apply all merges
                for merge in self.merges:
                    word_tokens = self.merge_tokens(word_tokens, merge)
                word_pairs = self.get_pairs(word_tokens)
                for pair in word_pairs:
                    pairs[pair] += freq
        
        # Build final vocabulary
        final_vocab = set()
        
        # Add special tokens first
        for token in self.special_tokens:
            final_vocab.add(token)
        
        # Add all possible subword units
        for word, freq in word_freqs.items():
            word_tokens = list(word[:-1]) + [word[-1] + '</w>']
            # Apply all merges
            for merge in self.merges:
                word_tokens = self.merge_tokens(word_tokens, merge)
            final_vocab.update(word_tokens)
        
        # Create vocab dictionaries
        self.vocab = {}
        self.idx2word = {}
        for i, token in enumerate(sorted(final_vocab)):
            self.vocab[token] = i
            self.idx2word[i] = token
        
        self.vocab_size = len(self.vocab)
        
        print(f"BPE training completed. Vocab size: {len(self.vocab)}")
        print(f"Number of merges: {len(self.merges)}")
    
    def encode(self, text, add_special_tokens=True):
        """Encode text into subword tokens"""
        if not isinstance(text, str):
            text = ""
        
        text = text.strip().lower()
        words = text.split()
        tokens = []
        
        for word in words:
            word_tokens = self.get_word_tokens(word)
            # Apply all merges
            for merge in self.merges:
                word_tokens = self.merge_tokens(word_tokens, merge)
            
            # Convert to indices
            for token in word_tokens:
                tokens.append(self.vocab.get(token, self.vocab['<unk>']))
        
        if add_special_tokens:
            tokens = [self.vocab['<sos>']] + tokens + [self.vocab['<eos>']]
        
        return tokens
    
    def decode(self, tokens):
        """Decode tokens back to text"""
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        
        words = []
        current_word = ""
        
        for token in tokens:
            if token in [self.vocab['<pad>'], self.vocab['<sos>']]:
                continue
            elif token == self.vocab['<eos>']:
                break
            else:
                token_str = self.idx2word.get(token, '<unk>')
                if token_str.endswith('</w>'):
                    # End of word
                    current_word += token_str[:-4]  # Remove </w>
                    if current_word:
                        words.append(current_word)
                    current_word = ""
                else:
                    # Continue current word
                    current_word += token_str
        
        # Handle remaining word
        if current_word:
            words.append(current_word)
        
        return ' '.join(words)
    
    def save(self, file_path):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'idx2word': self.idx2word,
            'merges': self.merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, file_path):
        """Load tokenizer from file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data['vocab']
        self.idx2word = data['idx2word']
        self.merges = data['merges']
        self.vocab_size = data['vocab_size']
        self.special_tokens = data['special_tokens']
    
    @property
    def word2idx(self):
        return self.vocab

# Improved Tokenizer class that uses BPE
class Tokenizer:
    def __init__(self, vocab_size=32000):
        self.bpe = BPETokenizer(vocab_size=vocab_size)
        self.vocab_size = vocab_size
    
    def build_vocab(self, sentences, min_freq=2):
        """Build vocabulary using BPE"""
        self.bpe.build_vocab(sentences, min_freq)
        self.vocab_size = self.bpe.vocab_size
    
    def encode(self, text, add_special_tokens=False):
        """Encode text to token indices"""
        return self.bpe.encode(text, add_special_tokens)
    
    def decode(self, tokens):
        """Decode token indices to text"""
        return self.bpe.decode(tokens)
    
    @property
    def word2idx(self):
        return self.bpe.vocab
    
    def save(self, file_path):
        self.bpe.save(file_path)
    
    def load(self, file_path):
        self.bpe.load(file_path)

def load_data(data_path, max_samples=None):
    """Load parallel data from file"""
    src_sentences = []
    tgt_sentences = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            line = line.strip()
            if '\t' in line:
                src_sent, tgt_sent = line.split('\t', 1)
                src_sentences.append(src_sent.strip())
                tgt_sentences.append(tgt_sent.strip())
    
    print(f"Loaded {len(src_sentences)} sentence pairs")
    return src_sentences, tgt_sentences

def split_data(src_sentences, tgt_sentences, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation, and test sets"""
    total_len = len(src_sentences)
    train_len = int(total_len * train_ratio)
    val_len = int(total_len * val_ratio)
    
    indices = list(range(total_len))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(indices)
    
    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len + val_len]
    test_indices = indices[train_len + val_len:]
    
    train_src = [src_sentences[i] for i in train_indices]
    train_tgt = [tgt_sentences[i] for i in train_indices]
    val_src = [src_sentences[i] for i in val_indices]
    val_tgt = [tgt_sentences[i] for i in val_indices]
    test_src = [src_sentences[i] for i in test_indices]
    test_tgt = [tgt_sentences[i] for i in test_indices]
    
    return (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt)

def calculate_bleu(references, predictions):
    """Calculate BLEU score (simplified implementation)"""
    def get_ngrams(sentence, n):
        words = sentence.split()
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    if not predictions or not references:
        return 0.0
    
    total_score = 0
    for pred, ref in zip(predictions, references):
        if not pred.strip() or not ref.strip():
            continue
            
        scores = []
        
        # Calculate precision for n-grams (n=1 to 4)
        for n in range(1, 5):
            pred_ngrams = Counter(get_ngrams(pred, n))
            ref_ngrams = Counter(get_ngrams(ref, n))
            
            if len(pred_ngrams) == 0:
                scores.append(0.0)
                continue
            
            # Calculate precision
            matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) 
                         for ngram in pred_ngrams)
            precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
            scores.append(precision)
        
        # Calculate geometric mean
        if all(score > 0 for score in scores):
            bleu = math.exp(sum(math.log(score) for score in scores) / 4)
        else:
            bleu = 0.0
        
        # Brevity penalty
        pred_len = len(pred.split())
        ref_len = len(ref.split())
        if pred_len < ref_len and pred_len > 0:
            bp = math.exp(1 - ref_len / pred_len)
        else:
            bp = 1.0
        
        total_score += bleu * bp
    
    return total_score / len(predictions) if predictions else 0.0

class Dataset:
    def __init__(self, src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, max_length=128):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Encode sentences
        src_tokens = self.src_tokenizer.encode(src_sentence, add_special_tokens=False)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_sentence, add_special_tokens=False)
        
        # Add SOS and EOS tokens
        src_tokens = [self.src_tokenizer.word2idx['<sos>']] + src_tokens + [self.src_tokenizer.word2idx['<eos>']]
        tgt_tokens = [self.tgt_tokenizer.word2idx['<sos>']] + tgt_tokens + [self.tgt_tokenizer.word2idx['<eos>']]
        
        # Pad sequences
        if len(src_tokens) < self.max_length:
            src_tokens.extend([self.src_tokenizer.word2idx['<pad>']] * (self.max_length - len(src_tokens)))
        else:
            src_tokens = src_tokens[:self.max_length-1] + [self.src_tokenizer.word2idx['<eos>']]
        
        if len(tgt_tokens) < self.max_length:
            tgt_tokens.extend([self.tgt_tokenizer.word2idx['<pad>']] * (self.max_length - len(tgt_tokens)))
        else:
            tgt_tokens = tgt_tokens[:self.max_length-1] + [self.tgt_tokenizer.word2idx['<eos>']]
        
        # Create target input (without last token) and target output (without first token)
        tgt_input = tgt_tokens[:-1]
        target = tgt_tokens[1:]
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_input, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

class LearningRateScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = max(1, step)  # Avoid division by zero
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']