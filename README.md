# Kokoro English TTS

Training implementation for English text-to-speech using the Kokoro Transformer architecture with LJSpeech dataset support.

## Current State

This is a simplified training implementation based on the Kokoro architecture. The official Kokoro-82M uses a decoder-only architecture based on StyleTTS 2 and iSTFTNet, with no diffusion or encoder. It employs a text encoder (phoneme-level BERT), style encoder for prosody control, WavLM-based discriminator (12 layers, pre-trained on 94k hours), and iSTFTNet vocoder which generates magnitude and phase for inverse STFT conversion rather than direct waveform generation. The training uses two stages: first training acoustic modules for mel-spectrogram reconstruction, then training TTS prediction modules with style diffusion and adversarial training.

This implementation differs significantly: uses explicit MFA-derived durations with a duration predictor instead of StyleTTS 2's diffusion-based style modeling and alignment, employs teacher forcing with standard multi-head attention rather than WavLM discriminator and adversarial training, has no style encoder or multi-speaker embeddings, uses a simple encoder-decoder transformer (~22M parameters vs 82M), and relies on external HiFi-GAN vocoder instead of integrated iSTFTNet. This implementation prioritizes training clarity and educational value over matching the production architecture.

## Features

Full encoder-decoder transformer with multi-head attention, phoneme-level duration alignment using Montreal Forced Aligner (MFA), CUDA support with optional mixed precision training, experiment tracking via Weights & Biases, checkpoint management for resuming training, and adaptive memory management with gradient checkpointing.

## Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
pip install "misaki[en]"
```

Download LJSpeech dataset with pre-aligned MFA annotations (3.8GB, recommended):
```bash
python setup_ljspeech.py --zenodo
```

Or download and align yourself (2.6GB + 1-3 hours):
```bash
python setup_ljspeech.py
python setup_ljspeech.py --align
```

Start training:
```bash
python training_english.py --corpus LJSpeech-1.1 --wandb
```

Resume from checkpoint:
```bash
python training_english.py --corpus LJSpeech-1.1 --resume auto --wandb
```

Generate speech:
```bash
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

The model consists of a text encoder (6-layer transformer with 8 attention heads), duration predictor (MLP for phoneme durations), length regulator (expands encoder outputs), mel decoder (6-layer transformer with masked attention), and stop token predictor.

Configuration: 512 hidden dimensions, 6 encoder/decoder layers, 8 attention heads, 2048 feed-forward dimensions, 80 mel channels, 22,050 Hz sample rate. Gradient checkpointing enabled for memory efficiency.

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

Setup on Paperspace:
```bash
cd /storage

wget https://zenodo.org/records/7499098/files/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2

pip install -r requirements.txt
pip uninstall -y spacy-curated-transformers
pip install transformers==4.35.2
pip install "numpy<2"
pip install "misaki[en]"
pip install --upgrade typing-extensions

wandb login

python training_english.py \
  --corpus /storage/LJSpeech-1.1 \
  --output /storage/kokoro_english_model \
  --batch-size 16 \
  --no-mixed-precision \
  --wandb
```

P4000 GPU (8GB) takes about 38 minutes per epoch with batch size 16. Full training (100 epochs) takes around 63 hours. Checkpoints are saved every 5 epochs at ~200MB each.

## Monitoring with Weights & Biases

Training with `--wandb` logs loss curves (total, mel, duration, stop token), learning rate schedule, system metrics (GPU utilization, memory, temperature), and memory management stats (cleanup frequency, pressure levels). Charts update every 10 batches.

## Inference

Basic usage:
```python
from inference_english import EnglishTTSInference

tts = EnglishTTSInference(
    model_path="kokoro_english_model/kokoro_english_final.pth",
    device="cuda"
)

tts.synthesize_to_file(
    text="Hello, how are you today?",
    output_path="output.wav"
)
```

Advanced options:
```bash
python inference_english.py \
  --model kokoro_english_model/checkpoint_epoch_50.pth \
  --text "Your text here" \
  --output output.wav \
  --device cuda \
  --vocoder hifigan
```

## Troubleshooting

`ImportError: cannot import name 'TypeIs' from 'typing_extensions'`
Run `pip install --upgrade typing-extensions`

Mixed precision errors on CUDA
Add `--no-mixed-precision` flag

Out of memory
Reduce `--batch-size` (try 8, 4, or 2)

W&B not showing loss charts
Fixed in latest version (losses log every 10 batches)

Performance tips: Use batch size 16-32 for CUDA GPUs. CPU training not recommended, but if needed use batch size 2-4. Gradient checkpointing is enabled by default. Pre-aligned Zenodo dataset saves 1-3 hours of setup.

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

Run implementation tests:
```bash
python test_english_implementation.py
```

Quick training test (100 samples):
```bash
python training_english.py --test-mode
```

## Model Outputs

Training generates checkpoints every 5 epochs, a phoneme processor file, and a final model. Each checkpoint contains model state dict, optimizer state, learning rate scheduler state, training configuration, current epoch and loss, and mixed precision scaler state.

## License

This implementation is for educational and research purposes.

## Acknowledgments

Based on the original Kokoro TTS model. LJSpeech dataset by Keith Ito. Montreal Forced Aligner for phoneme-level alignments. Misaki for English G2P. Original implementation based on [kokoro-ruslan](https://github.com/igorshmukler/kokoro-ruslan).
