import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pickle
import os
import tempfile
from collections import Counter

class SentencePieceTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.sp = None
        self.model_path = None
        
    def build_vocab(self, sentences, min_freq=2):
        print(f"Training SentencePiece BPE tokenizer with vocab size {self.vocab_size}...")
        
        try:
            import sentencepiece as spm
        except ImportError:
            print("ERROR: SentencePiece not installed!")
            print("Run: pip install sentencepiece")
            raise ImportError("Install sentencepiece: pip install sentencepiece")
        
        # Write sentences to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for sentence in sentences:
                if isinstance(sentence, str) and sentence.strip():
                    f.write(sentence.strip() + '\n')
            temp_file = f.name
        
        # Create model prefix
        model_prefix = tempfile.mktemp()
        
        try:
            # Train SentencePiece model - THIS IS FAST (1-2 minutes)
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=self.vocab_size,
                model_type='bpe',
                pad_id=0, unk_id=3, bos_id=1, eos_id=2,
                pad_piece='<pad>', unk_piece='<unk>', 
                bos_piece='<sos>', eos_piece='<eos>',
                character_coverage=0.9995,
                num_threads=8,  # Use multiple threads for speed
                split_by_unicode_script=True,
                shuffle_input_sentence=True,
                train_extremely_large_corpus=False
            )
            
            # Load the trained model
            self.sp = spm.SentencePieceProcessor(model_file=f'{model_prefix}.model')
            self.model_path = f'{model_prefix}.model'
            
            print(f"✅ SentencePiece BPE training completed!")
            print(f"   Actual vocab size: {self.sp.vocab_size()}")
            print(f"   Training time: ~1-2 minutes")
            
        except Exception as e:
            print(f"❌ SentencePiece training failed: {e}")
            raise
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def encode(self, text, add_special_tokens=False):
        if not self.sp:
            raise ValueError("Tokenizer not trained yet!")
        
        if not isinstance(text, str):
            text = ""
        
        # SentencePiece encode
        ids = self.sp.encode(text.strip())
        
        if add_special_tokens:
            # Add BOS and EOS
            ids = [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        
        return ids
    
    def decode(self, tokens):
        if not self.sp:
            raise ValueError("Tokenizer not trained yet!")
        
        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        
        # Convert to integers
        int_tokens = []
        for token in tokens:
            if hasattr(token, 'item'):
                token = token.item()
            int_tokens.append(int(token))
        
        # Decode
        text = self.sp.decode(int_tokens)
        return text.strip()
    
    @property
    def word2idx(self):
        if not self.sp:
            return {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        
        # Create vocab dict
        vocab = {}
        for i in range(self.sp.vocab_size()):
            piece = self.sp.id_to_piece(i)
            vocab[piece] = i
        return vocab
    
    def save(self, file_path):
        if not self.sp or not self.model_path:
            raise ValueError("No trained model to save!")
        
        # Copy the SentencePiece model file
        import shutil
        base_path = file_path.replace('.pkl', '')
        model_save_path = f"{base_path}.model"
        shutil.copy2(self.model_path, model_save_path)
        
        # Save metadata
        data = {
            'vocab_size': self.vocab_size,
            'model_path': model_save_path
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"SentencePiece model saved to {model_save_path}")
    
    def load(self, file_path):
        try:
            import sentencepiece as spm
            
            # Load metadata
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            self.vocab_size = data['vocab_size']
            model_path = data['model_path']
            
            if os.path.exists(model_path):
                self.sp = spm.SentencePieceProcessor(model_file=model_path)
                self.model_path = model_path
                print(f"SentencePiece model loaded from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
        except Exception as e:
            print(f"Failed to load SentencePiece model: {e}")
            raise

class Tokenizer:
    def __init__(self, vocab_size=32000):  # Full 32k vocab like your friend
        self.smp_tokenizer = SentencePieceTokenizer(vocab_size)
        self.vocab_size = vocab_size
    
    def build_vocab(self, sentences, min_freq=2):
        self.smp_tokenizer.build_vocab(sentences, min_freq)
        self.vocab_size = self.smp_tokenizer.sp.vocab_size()
    
    def encode(self, text, add_special_tokens=False):
        return self.smp_tokenizer.encode(text, add_special_tokens)
    
    def decode(self, tokens):
        return self.smp_tokenizer.decode(tokens)
    
    @property
    def word2idx(self):
        return self.smp_tokenizer.word2idx
    
    def save(self, file_path):
        self.smp_tokenizer.save(file_path)
    
    def load(self, file_path):
        self.smp_tokenizer.load(file_path)

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
    np.random.seed(42)
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
    """Calculate BLEU score"""
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
        for n in range(1, 5):
            pred_ngrams = Counter(get_ngrams(pred, n))
            ref_ngrams = Counter(get_ngrams(ref, n))
            
            if len(pred_ngrams) == 0:
                scores.append(0.0)
                continue
            
            matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) 
                         for ngram in pred_ngrams)
            precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
            scores.append(precision)
        
        if all(score > 0 for score in scores):
            bleu = math.exp(sum(math.log(score) for score in scores) / 4)
        else:
            bleu = 0.0
        
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
        
        # Encode sentences (SentencePiece handles special tokens)
        src_tokens = self.src_tokenizer.encode(src_sentence, add_special_tokens=True)
        tgt_tokens = self.tgt_tokenizer.encode(tgt_sentence, add_special_tokens=True)
        
        # Pad sequences
        if len(src_tokens) < self.max_length:
            src_tokens.extend([0] * (self.max_length - len(src_tokens)))  # 0 is pad_id
        else:
            src_tokens = src_tokens[:self.max_length-1] + [2]  # 2 is eos_id
        
        if len(tgt_tokens) < self.max_length:
            tgt_tokens.extend([0] * (self.max_length - len(tgt_tokens)))
        else:
            tgt_tokens = tgt_tokens[:self.max_length-1] + [2]
        
        # Create target input and target output
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
        step = max(1, step)
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