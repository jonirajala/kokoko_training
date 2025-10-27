# Kokoro Model Architecture

Core Transformer-based TTS model architecture.

## Differences from Kokoro-82M

The official Kokoro-82M is a decoder-only architecture based on StyleTTS 2 and iSTFTNet, trained in two stages with adversarial training. Key components include:

**Text Encoder**: Phoneme-level BERT transformer pre-trained on Wikipedia, encoding input phonemes into representations. StyleTTS 2 uses both acoustic and prosodic text encoders.

**Style Encoder**: For multi-speaker synthesis, extracts style vectors from reference audio to control prosody and speaker characteristics. This enables zero-shot voice cloning and style transfer.

**Discriminator**: 12-layer WavLM model pre-trained on 94k hours of speech data, frozen during training to prevent overpowering. Used in adversarial training to improve naturalness.

**iSTFTNet Vocoder**: Instead of directly generating waveforms like HiFi-GAN, it predicts magnitude and phase spectrograms which are converted to audio via inverse STFT. This hybrid approach reduces computational cost and model size while maintaining quality.

**Training**: Two-stage process. Stage 1 trains acoustic modules for mel-spectrogram reconstruction. Stage 2 trains TTS prediction modules (duration, prosody) using fixed acoustic modules from stage 1, with style diffusion and adversarial training.

This implementation differs fundamentally: uses a simple encoder-decoder transformer (~22M vs 82M parameters), explicit MFA-derived durations instead of learned alignments through diffusion, teacher forcing with standard multi-head attention instead of WavLM adversarial training, no style encoder (no prosody control or multi-speaker support), single-stage training without adversarial loss, and external HiFi-GAN vocoder instead of integrated iSTFTNet. This version prioritizes educational clarity over production sophistication.

| Component | Kokoro-82M (Official) | This Implementation |
|-----------|----------------------|---------------------|
| **Architecture Type** | Decoder-only | Encoder-decoder |
| **Base Model** | StyleTTS 2 + iSTFTNet | Custom transformer |
| **Parameters** | 82M | ~22M |
| **Text Encoding** | BERT (phoneme-level, pre-trained) | Transformer encoder (6 layers) |
| **Style Modeling** | Style encoder + diffusion | None |
| **Duration Modeling** | Learned via alignment + diffusion | Explicit MLP predictor (MFA) |
| **Prosody Control** | Style vectors from reference audio | None |
| **Speaker Control** | Multi-speaker (zero-shot cloning) | Single speaker only |
| **Discriminator** | WavLM (12 layers, frozen, 94k hours) | None |
| **Training Stages** | Two-stage (acoustic → TTS) | Single-stage |
| **Training Method** | Adversarial + diffusion | Supervised (MSE + BCE) |
| **Vocoder** | Integrated iSTFTNet (mag + phase) | External HiFi-GAN |
| **Attention Type** | StyleTTS 2 attention mechanisms | Standard multi-head |
| **Training Data** | Few hundred hours (permissive) | LJSpeech (24 hours) |
| **Output** | 24kHz audio | 22.05kHz audio |

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
