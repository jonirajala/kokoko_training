# Kokoro Model Architecture

Core Transformer-based TTS model architecture.

## Differences from Kokoro-82M

This implementation differs from the official Kokoro-82M released model in key ways. The official model uses monotonic alignment search (similar to Tacotron 2) for learning alignments during training, while this uses explicit MFA-derived durations with a duration predictor. Kokoro-82M employs guided attention mechanisms and location-sensitive attention, whereas this uses standard multi-head attention with teacher forcing. The production model includes style tokens for prosody control and multi-speaker embeddings, which are absent here. Additionally, Kokoro-82M has various production optimizations (custom CUDA kernels, quantization support, streaming inference) not present in this educational implementation. This version is ~22M parameters vs the 82M in the official release, using a simpler architecture focused on training transparency.

## Files

`model.py` contains the complete Kokoro TTS model with text encoder (Transformer), duration predictor (MLP), length regulator (duration expansion), mel decoder (Transformer), and stop token predictor. Main methods are `forward()` for training with teacher forcing, `inference()` for autoregressive generation, and `get_model_info()` for parameter stats.

`model_transformers.py` implements the transformer encoder and decoder blocks with multi-head self-attention and gradient checkpointing support.

`positional_encoding.py` provides sinusoidal positional encoding for sequence order (fixed, not learned), supporting sequences up to max_len (default 5000) with dropout regularization.

## Architecture Overview

```
Text → Encoder → Duration → Length → Decoder → Mel Spectrogram
                 Predictor   Regulator         + Stop Token
```

Training flow uses teacher forcing: text becomes phoneme indices, encoder processes phoneme sequence, duration predictor predicts phoneme durations, length regulator expands encoder outputs, decoder generates mel frames using ground truth as input, outputting mel spectrogram and stop tokens.

Inference flow is autoregressive: same initial steps, but decoder generates mel frames step-by-step until stop token threshold is reached.

## Model Parameters

Default configuration: 63 vocab size (English phonemes), 512 hidden dim, 6 encoder layers, 6 decoder layers, 8 attention heads, 2048 feed-forward dim, 80 mel channels, gradient checkpointing with 4 segments.

Total parameters: ~5.7M (small) or ~22M (default). Model size: ~22 MB (small) or ~85 MB (default).

Clean separation of encoder, duration, and decoder. Gradient checkpointing for large batches. Configurable layers, heads, and dimensions. Supports both training and inference modes. Uses teacher forcing during training, proper autoregressive causal masking, handles variable length sequences, explicit phoneme duration prediction. Checkpoint segments reduce memory by ~75%.

## Usage

```python
from kokoro.model import KokoroModel

model = KokoroModel(
    vocab_size=63,
    mel_dim=80,
    hidden_dim=512,
    n_encoder_layers=6,
    n_decoder_layers=6
)

# Training
mel_pred, duration_pred, stop_pred = model(
    phoneme_indices,
    mel_specs,
    phoneme_durations,
    stop_token_targets
)

# Inference
mel_output = model.inference(phoneme_indices, max_mel_len=1000)
```

Requires PyTorch 2.0+. Gradient checkpointing trades compute for memory. GPUProfiler is a lightweight stub. Supports CUDA, MPS (Apple Silicon), and CPU.
