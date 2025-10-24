# Kokoro English TTS

A Text-to-Speech (TTS) training implementation for English using the Kokoro Transformer architecture with LJSpeech dataset support.

## Features

- **Modern Transformer Architecture**: Full encoder-decoder with multi-head attention
- **Montreal Forced Aligner (MFA)**: Phoneme-level duration alignment for high-quality synthesis
- **GPU Optimized**: CUDA support with optional mixed precision training
- **Weights & Biases**: Built-in experiment tracking and monitoring
- **Checkpoint Management**: Resume training from any saved checkpoint
- **Memory Efficient**: Adaptive memory management and gradient checkpointing

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install Misaki for English G2P
pip install "misaki[en]"
```

### 2. Download LJSpeech Dataset

```bash
# Download with pre-aligned MFA annotations (recommended - 3.8GB)
python setup_ljspeech.py --zenodo

# Or download original and align yourself (2.6GB + 1-3 hours alignment)
python setup_ljspeech.py
python setup_ljspeech.py --align
```

### 3. Start Training

```bash
# Basic training
python training_english.py --corpus LJSpeech-1.1 --wandb

# Full configuration
python training_english.py \
  --corpus LJSpeech-1.1 \
  --output kokoro_english_model \
  --batch-size 16 \
  --epochs 100 \
  --wandb

# Resume from checkpoint
python training_english.py \
  --corpus LJSpeech-1.1 \
  --resume auto \
  --wandb
```

### 4. Generate Speech

```bash
# Run inference
python inference_english.py \
  --model kokoro_english_model/kokoro_english_final.pth \
  --text "Hello world, this is a test." \
  --output output.wav
```

## Training Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--corpus` | `./LJSpeech-1.1` | Path to LJSpeech dataset |
| `--output` | `./kokoro_english_model` | Output directory for checkpoints |
| `--batch-size` | `8` | Training batch size |
| `--epochs` | `100` | Number of training epochs |
| `--learning-rate` | `1e-4` | Learning rate |
| `--resume` | `None` | Resume from checkpoint (auto or path) |
| `--wandb` | `False` | Enable Weights & Biases logging |
| `--no-mixed-precision` | `False` | Disable mixed precision training |
| `--test-mode` | `False` | Quick test with 100 samples |

## Model Architecture

### Core Components

- **Text Encoder**: 6-layer Transformer encoder with 8 attention heads
- **Duration Predictor**: MLP predicting phoneme durations
- **Length Regulator**: Expands encoder outputs based on predicted durations
- **Mel Decoder**: 6-layer Transformer decoder with masked attention
- **Stop Token Predictor**: Predicts end-of-sequence

### Configuration

- **Hidden Dimension**: 512
- **Encoder/Decoder Layers**: 6 each
- **Attention Heads**: 8
- **Feed-Forward Dimension**: 2048
- **Mel Channels**: 80
- **Sample Rate**: 22,050 Hz
- **Gradient Checkpointing**: Enabled for memory efficiency

## Dataset Structure

The LJSpeech dataset should be organized as:

```
LJSpeech-1.1/
├── metadata.csv           # Transcriptions
├── wavs/                  # Audio files (13,100 samples)
│   ├── LJ001-0001.wav
│   └── ...
└── TextGrid/              # MFA alignments (if using Zenodo)
    ├── LJ001-0001.TextGrid
    └── ...
```

## Cloud Training (Paperspace/Colab)

### Setup on Paperspace

```bash
# Navigate to persistent storage
cd /storage

# Download dataset directly on cloud
wget https://zenodo.org/records/7499098/files/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

# Install dependencies
pip install -r requirements.txt
pip uninstall -y spacy-curated-transformers
pip install transformers==4.35.2
pip install "numpy<2"
pip install "misaki[en]"
pip install --upgrade typing-extensions

# Login to W&B
wandb login

# Start training
python training_english.py \
  --corpus /storage/LJSpeech-1.1 \
  --output /storage/kokoro_english_model \
  --batch-size 16 \
  --no-mixed-precision \
  --wandb
```

### Performance Expectations

- **P4000 GPU (8GB)**: ~38 minutes per epoch, batch size 16
- **Expected Training**: 100 epochs = ~63 hours
- **Checkpoints**: Saved every 5 epochs (~200MB each)

## Monitoring with Weights & Biases

When training with `--wandb`, you'll see:

- **Loss curves**: Total, mel, duration, stop token losses
- **Learning rate**: Schedule over time
- **System metrics**: GPU utilization, memory, temperature
- **Memory management**: Cleanup frequency, pressure levels

Charts update every 10 batches for smooth visualization.

## Inference

### Basic Inference

```python
from inference_english import EnglishTTSInference

# Load model
tts = EnglishTTSInference(
    model_path="kokoro_english_model/kokoro_english_final.pth",
    device="cuda"
)

# Generate speech
tts.synthesize_to_file(
    text="Hello, how are you today?",
    output_path="output.wav"
)
```

### Advanced Options

```bash
python inference_english.py \
  --model kokoro_english_model/checkpoint_epoch_50.pth \
  --text "Your text here" \
  --output output.wav \
  --device cuda \
  --vocoder hifigan  # or 'griffin-lim'
```

## Troubleshooting

### Common Issues

**Problem**: `ImportError: cannot import name 'TypeIs' from 'typing_extensions'`
**Solution**: `pip install --upgrade typing-extensions`

**Problem**: Mixed precision errors on CUDA
**Solution**: Add `--no-mixed-precision` flag

**Problem**: Out of memory
**Solution**: Reduce `--batch-size` (try 8, 4, or 2)

**Problem**: W&B not showing loss charts
**Solution**: This was fixed in the latest version - losses now log every 10 batches

### Performance Tips

- **GPU Training**: Use batch size 16-32 for CUDA GPUs
- **CPU Training**: Not recommended, use batch size 2-4 if necessary
- **Memory**: Gradient checkpointing is enabled by default
- **Dataset**: Pre-aligned Zenodo version saves 1-3 hours of setup time

## File Structure

```
kokoro-english-tts/
├── README.md
├── requirements.txt
├── setup_ljspeech.py                # Dataset setup
├── training_english.py              # Main training script
├── inference_english.py             # Main inference script
├── test_english_implementation.py   # Test suite
│
├── kokoro/                          # Core model architecture
│   ├── __init__.py
│   ├── model.py                     # Kokoro TTS model
│   ├── model_transformers.py        # Transformer encoder/decoder
│   └── positional_encoding.py      # Sinusoidal encoding
│
├── data/                            # Dataset and preprocessing
│   ├── __init__.py
│   ├── ljspeech_dataset.py          # LJSpeech data loader
│   └── english_phoneme_processor.py # English G2P (Misaki)
│
├── audio/                           # Audio processing and vocoder
│   ├── __init__.py
│   ├── audio_utils.py               # Audio utilities
│   ├── vocoder_manager.py           # Vocoder interface
│   └── hifigan_vocoder.py           # HiFi-GAN implementation
│
└── training/                        # Training infrastructure
    ├── __init__.py
    ├── config_english.py            # Training configuration
    ├── trainer.py                   # Base trainer
    ├── english_trainer.py           # English trainer with W&B
    ├── checkpoint_manager.py        # Checkpoint utilities
    ├── adaptive_memory_manager.py   # Memory optimization
    ├── interbatch_profiler.py       # Performance profiling
    ├── mps_grad_scaler.py           # MPS mixed precision
    └── device_type.py               # Device enumeration
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA 11.8+ for GPU)
- misaki[en] - English G2P
- librosa, soundfile - Audio processing
- wandb - Experiment tracking (optional)
- tqdm - Progress bars

See `requirements.txt` for full list.

## Testing

```bash
# Run implementation tests
python test_english_implementation.py

# Quick training test (100 samples)
python training_english.py --test-mode
```

## Model Outputs

Training generates:

```
kokoro_english_model/
├── checkpoint_epoch_5.pth       # Regular checkpoints
├── checkpoint_epoch_10.pth
├── ...
├── phoneme_processor.pkl        # English phoneme processor
└── kokoro_english_final.pth     # Final trained model
```

Each checkpoint contains:
- Model state dict
- Optimizer state
- Learning rate scheduler state
- Training configuration
- Current epoch and loss
- Mixed precision scaler state

## License

This implementation is for educational and research purposes.

## Acknowledgments

- **Kokoro Architecture**: Based on the original Kokoro TTS model
- **LJSpeech Dataset**: Public domain speech corpus by Keith Ito
- **Montreal Forced Aligner**: For phoneme-level alignments
- **Misaki**: English grapheme-to-phoneme conversion
- **Credits**: Original implementation based on [kokoro-ruslan](https://github.com/igorshmukler/kokoro-ruslan)

## Contributing

Contributions welcome! Please open an issue or PR.
