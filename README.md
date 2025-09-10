# Transformer from Scratch - Finnish to English Machine Translation

This project implements a complete Transformer architecture from scratch for Finnish-English machine translation, as part of the Advanced NLP course assignment.

## Features

- **Complete Transformer Architecture**: Encoder-decoder model with multi-head attention implemented from scratch
- **Two Positional Encoding Methods**:
  - Rotary Positional Embeddings (RoPE)
  - Relative Position Bias (additive bias to attention scores)
- **Three Decoding Strategies**:
  - Greedy Decoding
  - Beam Search
  - Top-k Sampling
- **Full Training Pipeline**: With teacher forcing, checkpointing, and learning rate scheduling
- **Comprehensive Evaluation**: BLEU score calculation and comparative analysis

## Project Structure

```
├── encoder.py          # Encoder implementation with attention mechanisms
├── decoder.py          # Decoder implementation and decoding strategies
├── train.py           # Training script with full pipeline
├── test.py            # Testing and evaluation script
├── utils.py           # Utility functions (tokenizer, dataset, metrics)
├── README.md          # This file
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install torch torchvision matplotlib tqdm numpy
```

## Dataset Preparation

The dataset should be a tab-separated file with Finnish sentences in the first column and English sentences in the second column:

```
Hei maailma	Hello world
Kuinka voit?	How are you?
```

## Usage

### Training

Train a model with RoPE positional encoding:
```bash
python train.py --data_path your_data.txt --positional_encoding rope --epochs 50 --batch_size 32
```

Train a model with Relative Position Bias:
```bash
python train.py --data_path your_data.txt --positional_encoding relative --epochs 50 --batch_size 32
```

### Key Training Parameters

- `--d_model`: Model dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_encoder_layers`: Number of encoder layers (default: 6)
- `--num_decoder_layers`: Number of decoder layers (default: 6)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--positional_encoding`: Choose between 'rope' or 'relative'
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--max_length`: Maximum sequence length (default: 128)

### Testing

Test all decoding strategies:
```bash
python test.py --model_path ./models/best_model.pt --model_dir ./models --data_path your_data.txt --decoding_strategy all
```

Test specific decoding strategy:
```bash
python test.py --model_path ./models/best_model.pt --model_dir ./models --data_path your_data.txt --decoding_strategy greedy
```

# Transformer from Scratch - Finnish to English Machine Translation

This project implements a complete Transformer architecture from scratch for Finnish-English machine translation, as part of the Advanced NLP course assignment.

## Features

- **Complete Transformer Architecture**: Encoder-decoder model with multi-head attention implemented from scratch
- **Two Positional Encoding Methods**:
  - Rotary Positional Embeddings (RoPE)
  - Relative Position Bias (additive bias to attention scores)
- **Three Decoding Strategies**:
  - Greedy Decoding
  - Beam Search
  - Top-k Sampling
- **Full Training Pipeline**: With teacher forcing, checkpointing, and learning rate scheduling
- **Comprehensive Evaluation**: BLEU score calculation and comparative analysis

## Project Structure

```
├── encoder.py          # Encoder implementation with attention mechanisms
├── decoder.py          # Decoder implementation and decoding strategies
├── train.py           # Training script with full pipeline
├── test.py            # Testing and evaluation script
├── utils.py           # Utility functions (tokenizer, dataset, metrics)
├── prepare_dataset.py # Dataset preparation script
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Installation

1. Install required packages:
```bash
pip install numpy matplotlib tqdm scikit-learn psutil seaborn tensorboard
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Test GPU setup:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## Dataset Preparation

1. Place your dataset files (`EUbookshop.fi` and `EUbookshop.en`) in the project directory
2. Run the dataset preparation script:
```bash
python prepare_dataset.py
```
This creates `finnish_english_100k.txt` with 100K sentence pairs.

## Usage

### Complete Training Pipeline

**Step 1: Train RoPE Model (75-90 minutes)**
```bash
python train.py --data_path finnish_english_100k.txt --positional_encoding rope --batch_size 16 --epochs 15 --mixed_precision --model_dir ./models/rope_model --log_dir ./logs/rope_logs
```

**Step 2: Train Relative Position Model (75-90 minutes)**
```bash
python train.py --data_path finnish_english_100k.txt --positional_encoding relative --batch_size 16 --epochs 15 --mixed_precision --model_dir ./models/relative_model --log_dir ./logs/relative_logs
```

**Step 3: Evaluate RoPE Model (20-30 minutes)**
```bash
mkdir -p results/rope_results
python test.py --model_path ./models/rope_model/best_model.pt --model_dir ./models/rope_model --data_path finnish_english_100k.txt --positional_encoding rope --decoding_strategy all --batch_size 16 --output_dir ./results/rope_results --detailed_analysis
```

**Step 4: Evaluate Relative Position Model (20-30 minutes)**
```bash
mkdir -p results/relative_results
python test.py --model_path ./models/relative_model/best_model.pt --model_dir ./models/relative_model --data_path finnish_english_100k.txt --positional_encoding relative --decoding_strategy all --batch_size 16 --output_dir ./results/relative_results --detailed_analysis
```

### Key Training Parameters

- `--d_model`: Model dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_encoder_layers`: Number of encoder layers (default: 6)
- `--num_decoder_layers`: Number of decoder layers (default: 6)
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--positional_encoding`: Choose between 'rope' or 'relative'
- `--batch_size`: Training batch size (optimized for RTX 4060: 16)
- `--mixed_precision`: Enable mixed precision training for memory efficiency
- `--epochs`: Number of training epochs (recommended: 15-20)

### Hardware Requirements

**Minimum:**
- RTX 4060 (8GB VRAM) or equivalent
- 16GB RAM
- ~4 hours total training time

**Optimal:**
- RTX 4070+ (12GB+ VRAM) 
- 32GB RAM
- Allows larger batch sizes and faster training

## Model Architecture Details

### Encoder
- Multi-head self-attention with configurable positional encoding
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Dropout for regularization

### Decoder
- Masked multi-head self-attention
- Multi-head cross-attention with encoder outputs
- Position-wise feed-forward networks
- Layer normalization and residual connections

### Positional Encoding

**Rotary Positional Embeddings (RoPE)**:
- Applies rotational transformations to query and key vectors
- Better handling of relative positions
- More effective for longer sequences

**Relative Position Bias**:
- Learnable bias terms added to attention scores
- Directly models relative distances between tokens
- Separate bias for each attention head

### Decoding Strategies

**Greedy Decoding**:
- Selects token with highest probability at each step
- Fastest but may miss better sequences
- Deterministic output

**Beam Search**:
- Maintains top-B candidate sequences
- Explores multiple possibilities simultaneously
- Better quality than greedy, slower than greedy

**Top-k Sampling**:
- Samples from top-k most probable tokens
- Introduces controlled randomness
- More diverse outputs than deterministic methods

## Expected Results

With 100K Finnish-English sentence pairs and proper training:

- **BLEU Scores**: 0.15-0.35 depending on model and decoding strategy
- **Training Loss**: Should decrease from ~10 to ~3-4 over 15 epochs
- **Convergence**: Clear learning curves showing which positional encoding works better
- **Translation Quality**: Readable Finnish-English translations

## File Outputs

### Training Outputs
- `./models/rope_model/best_model.pt`: Best RoPE model checkpoint
- `./models/relative_model/best_model.pt`: Best Relative Position model checkpoint
- `./models/*/src_tokenizer.pkl`: Source language tokenizer
- `./models/*/tgt_tokenizer.pkl`: Target language tokenizer
- `./logs/*/training_history.json`: Training metrics and curves

### Testing Outputs
- `./results/rope_results/`: Complete evaluation results for RoPE model
- `./results/relative_results/`: Complete evaluation results for Relative Position model
- `./results/final_comparison.png`: Side-by-side comparison plots

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
# Reduce batch size
python train.py --batch_size 8 --mixed_precision

# Or reduce model size
python train.py --batch_size 16 --d_model 384 --mixed_precision
```

**Slow Training:**
- Ensure CUDA is properly installed and detected
- Use `--mixed_precision` flag
- Monitor GPU utilization with `nvidia-smi`

**Poor BLEU Scores (<0.1):**
- Check data quality and preprocessing
- Increase training epochs (20-30)
- Verify model architecture parameters

## Implementation Notes

### Key Components Implemented from Scratch

1. **Multi-Head Attention**: Complete scaled dot-product attention with cross-attention support
2. **Positional Encodings**: Both RoPE and relative position bias
3. **Layer Normalization**: Applied in pre-norm configuration
4. **Feed-Forward Networks**: Position-wise transformations with ReLU activation
5. **Masking**: Padding masks and causal masks for decoder
6. **Beam Search**: Full beam search implementation with proper sequence ranking

### Performance Optimizations

- **Mixed precision training**: ~40% memory reduction, 30% speedup
- **Gradient clipping**: Training stability (max_norm=1.0)
- **Learning rate warmup**: Better convergence with cosine scheduling
- **Efficient data loading**: Multi-worker data loading with prefetching
- **Memory management**: Periodic cache clearing during training

## Assignment Deliverables

This implementation satisfies all assignment requirements:

✅ **Complete from-scratch implementation** (no PyTorch transformer modules)  
✅ **Two positional encoding methods** with easy switching  
✅ **Three decoding strategies** with comparative analysis  
✅ **Full training pipeline** with teacher forcing  
✅ **BLEU evaluation** and convergence analysis  
✅ **Proper file structure** and documentation  

## Total Time Investment

- **Setup and data preparation**: 30 minutes
- **RoPE model training**: 75-90 minutes  
- **Relative model training**: 75-90 minutes
- **Evaluation of both models**: 60 minutes
- **Analysis and report writing**: 60 minutes
- **Total**: 5-6 hours

## Pre-trained Models

After training, your best models will be saved as:
- `./models/rope_model/best_model.pt`
- `./models/relative_model/best_model.pt`

These contain the complete model state, optimizer state, and training metadata needed for evaluation and inference.

## References

1. Vaswani et al. "Attention is All You Need" (2017)
2. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
3. Shaw et al. "Self-Attention with Relative Position Representations" (2018)

python train.py --data_path "finnish_english_100k.txt" --positional_encoding relative --batch_size 32 --epochs 10 --learning_rate 1e-4 --model_dir ./models_relative --log_dir ./logs_relative --num_workers 0