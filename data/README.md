# Data Processing

Dataset loading and phoneme processing for English TTS.

## Files

`ljspeech_dataset.py` loads and processes the LJSpeech dataset (13,100 English speech samples) with MFA alignments. Includes `LJSpeechDataset` for data loading, `LengthBasedBatchSampler` for smart batching by sequence length, and `collate_fn` for batch padding. Reads MFA TextGrid alignments for duration, converts audio to mel spectrograms, handles variable-length sequences. Batching by length reduces padding by ~30-40%.

`english_phoneme_processor.py` converts English text to phonemes using Misaki. Vocabulary of 63 phonemes (IPA + special tokens): vowels (ə, ɪ, ɛ, æ, ʌ, ɔ, ʊ, i, u), consonants (p, b, t, d, k, g, f, v, s, z), special tokens (SIL for silence, SPN for spoken noise), and stress markers (ˈ primary, ˌ secondary). Has fallback mode if Misaki not installed.

## Dataset Format

LJSpeech structure:
```
LJSpeech-1.1/
├── metadata.csv          # id|transcription
├── wavs/
│   ├── LJ001-0001.wav   # 22050 Hz mono
│   └── ...
└── TextGrid/            # MFA alignments
    ├── LJ001-0001.TextGrid
    └── ...
```

TextGrid files contain phoneme-level alignments with start/end times. Used to extract phoneme durations in mel frames. If missing, uses uniform duration fallback (lower quality).

## Processing Pipeline

Training sample creation: load audio WAV (22050 Hz), compute 80-channel mel spectrogram, get transcription from metadata.csv, convert text to phonemes via Misaki, extract durations from TextGrid (or uniform), generate stop token targets (1 at end).

## Usage

```python
from data import LJSpeechDataset, LengthBasedBatchSampler, collate_fn
from training.config_english import get_default_config

config = get_default_config()
dataset = LJSpeechDataset("LJSpeech-1.1", config)

sampler = LengthBasedBatchSampler(dataset, batch_size=16, shuffle=True)
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)

for batch in loader:
    mel_specs = batch['mel_specs']  # [B, T, 80]
    phonemes = batch['phoneme_indices']  # [B, N]
    break
```

MFA alignments essential for quality (90% improvement over uniform fallback). Typical batch size 16-32 with length-based sampling. Dataset uses ~4GB memory with caching, processes ~500 samples/sec on modern CPU.
