# Kokoro Model Architecture

This folder contains the core Transformer-based TTS model architecture.

## Files

### `model.py` (Main Model)
**Purpose**: Complete Kokoro TTS model implementation
**Key Components**:
- `KokoroModel` - Main model class combining all components
- `GPUProfiler` - Lightweight memory tracking stub
- Text encoder (Transformer)
- Duration predictor (MLP)
- Length regulator (duration-based expansion)
- Mel decoder (Transformer)
- Stop token predictor

**Key Methods**:
- `forward()` - Main forward pass (training mode with teacher forcing)
- `inference()` - Autoregressive generation for synthesis
- `get_model_info()` - Model statistics and parameter count

**Dependencies**:
- `model_transformers.py` - Transformer encoder/decoder blocks
- `positional_encoding.py` - Sinusoidal position embeddings

### `model_transformers.py` (Transformer Layers)
**Purpose**: Transformer encoder and decoder implementations
**Components**:
- `TransformerEncoderBlock` - Multi-head self-attention encoder
- `TransformerDecoder` - Causal decoder with cross-attention
- Supports gradient checkpointing for memory efficiency

### `positional_encoding.py` (Position Embeddings)
**Purpose**: Sinusoidal positional encoding for sequence order
**Features**:
- Fixed sinusoidal embeddings (not learned)
- Supports sequences up to max_len (default 5000)
- Dropout for regularization

## Architecture Overview

```
Text → Encoder → Duration → Length → Decoder → Mel Spectrogram
                 Predictor   Regulator         + Stop Token
```

### Training Flow (Teacher Forcing):
1. Text → Phoneme indices
2. Encoder: Process phoneme sequence
3. Duration Predictor: Predict phoneme durations
4. Length Regulator: Expand encoder outputs
5. Decoder: Generate mel frames (with ground truth as input)
6. Outputs: Mel spectrogram + Stop tokens

### Inference Flow (Autoregressive):
1. Text → Phoneme indices
2. Encoder: Process phoneme sequence
3. Duration Predictor: Predict phoneme durations
4. Length Regulator: Expand encoder outputs
5. Decoder: Generate mel frames step-by-step
6. Stop when stop token threshold reached

## Model Parameters

**Default Configuration**:
- Vocab size: 63 (English phonemes)
- Hidden dim: 512
- Encoder layers: 6
- Decoder layers: 6
- Attention heads: 8
- Feed-forward dim: 2048 (encoder & decoder)
- Mel channels: 80
- Gradient checkpointing: Enabled (4 segments)

**Total Parameters**: ~5.7M (small model), ~22M (default)
**Model Size**: ~22 MB (small), ~85 MB (default)

## Design Principles

1. **Simplicity**: Clean separation of encoder, duration, and decoder
2. **Memory Efficiency**: Gradient checkpointing for large batches
3. **Flexibility**: Configurable layers, heads, and dimensions
4. **Production Ready**: Supports both training and inference modes

## Key Features

- **Teacher Forcing**: Uses ground truth during training
- **Causal Masking**: Proper autoregressive decoder
- **Variable Length**: Handles sequences of different lengths
- **Duration Modeling**: Explicit phoneme duration prediction
- **Memory Optimized**: Checkpoint segments reduce memory 75%

## Usage

```python
from kokoro.model import KokoroModel

# Create model
model = KokoroModel(
    vocab_size=63,
    mel_dim=80,
    hidden_dim=512,
    n_encoder_layers=6,
    n_decoder_layers=6
)

# Training
outputs = model(
    phoneme_indices,
    mel_specs,
    phoneme_durations,
    stop_token_targets
)
mel_pred, duration_pred, stop_pred = outputs

# Inference
mel_output = model.inference(
    phoneme_indices,
    max_mel_len=1000
)
```

## Notes

- Model uses PyTorch 2.0+ features
- Gradient checkpointing trades compute for memory
- GPUProfiler is a lightweight stub (full profiler was removed for simplicity)
- Supports CUDA, MPS (Apple Silicon), and CPU
