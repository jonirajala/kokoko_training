# TTS Inference Debugging Guide

After training for 100 epochs with low loss but getting garbage audio, use this guide to diagnose the problem.

## Quick Test

```bash
# Test inference with debugging enabled
python test_inference_debug.py --model ./kokoro_english_model --text "Hello world" --output test.wav
```

Or use the main inference script:

```bash
# Basic inference with debug mode
python inference_english.py --model ./kokoro_english_model --text "Hello world" --output test.wav --debug

# Interactive mode with debugging
python inference_english.py --model ./kokoro_english_model --interactive --debug
```

## What to Look For

The debug output will show you 4 critical stages:

### Stage 1: Text → Phonemes
```
✓ Generated 9 phonemes
Phonemes: HH AH0 L OW1 W ER1 L D
```

**Red flags:**
- ⚠ Unknown phonemes (<unk>) - means vocab mismatch
- Very few phonemes for long text
- Phonemes look wrong (not ARPA format)

### Stage 2: Phonemes → Indices
```
✓ Converted to 9 indices
Index range: [3, 89]
Vocab size: 87
```

**Red flags:**
- Index out of range (> vocab size)
- All indices are 1 (<unk>)
- Vocab size mismatch with training

### Stage 3: Model Inference (Critical!)
```
✓ Generated mel spectrogram
Mel shape: torch.Size([80, 123]) (channels=80, frames=123)
Mel range: [-11.234, -0.123]
Mel mean: -5.678, std: 2.345
```

**This is where problems usually show up:**

#### ❌ Problem: Mel values out of range
```
Mel range: [-15.0, 5.0]  # WRONG! Should be ~[-11.5, 0.0]
```
**Cause:** Model not properly trained, or inference mode not working
**Fix:** Check if duration clamping is applied, retrain if needed

#### ❌ Problem: Mel is constant/zero variance
```
Mel std: 0.0001  # WRONG! Should be > 1.0
❌ CRITICAL: Mel spectrogram is nearly constant!
```
**Cause:** Model producing flat output, not learning properly
**Fix:** Training issue - model collapsed or not converging

#### ❌ Problem: NaN or Inf values
```
❌ CRITICAL: Mel spectrogram contains NaN values!
```
**Cause:** Gradient explosion during training, or inference numerical instability
**Fix:** Check duration predictor clamping, gradient clipping during training

### Stage 4: Vocoder (Mel → Audio)
```
✓ Generated audio
Audio duration: 1.23s
Audio range: [-0.45, 0.52]
```

**Red flags:**
- Audio amplitude < 0.001 (silent)
- Audio std < 0.001 (no variation, garbage)
- NaN or Inf in audio

## Common Problems & Solutions

### Problem 1: Model outputs constant mel spectrogram
**Symptoms:**
- Mel std < 0.1
- Audio is noise/garbage
- Loss was low during training (0.4)

**Diagnosis:**
The model learned to output a "safe" constant value that minimizes loss without actually learning the mapping.

**Causes:**
1. Duration predictor not working (using uniform durations)
2. Alignment mismatch during training (MFA vs G2P)
3. Mel spectrogram normalization wrong during training
4. Model capacity too small

**Solutions:**
1. Check alignment usage: Were 100% of samples using MFA or falling back to uniform?
2. Verify mel range during training was correct ([-11.5, 0.0])
3. Check duration predictor logs - was it learning reasonable durations?
4. Try larger model or more training epochs

### Problem 2: Mel values completely wrong range
**Symptoms:**
- Mel range like [-20, 10] instead of [-11.5, 0.0]
- Loss was reasonable but audio is garbage

**Diagnosis:**
Model learned on wrong mel range, or inference clamping not matching training.

**Solutions:**
1. Check `data/ljspeech_dataset.py` line 335 - mel clamping during training
2. Check `kokoro/model.py` - is inference applying same normalization?
3. Retrain with correct mel range

### Problem 3: Unknown phonemes during inference
**Symptoms:**
- `⚠ Found 5 unknown phonemes (<unk>)!`
- Phoneme vocab size different from training

**Diagnosis:**
Using different phoneme processor at inference vs training.

**Solutions:**
1. Check `phoneme_processor.pkl` was saved correctly during training
2. Verify g2p_en vs Misaki - must match training
3. Regenerate model with correct processor

### Problem 4: Duration predictor producing extreme values
**Symptoms:**
- Some phonemes get 1000+ frames
- Audio is VERY slow or cuts off early
- Mel shape is huge (> 1000 frames)

**Diagnosis:**
Duration predictor not clamped during inference.

**Solutions:**
1. Check `kokoro/model.py:304` - log duration clamping in `forward_inference()`
2. Ensure same clamping as training: `torch.clamp(log_durations, min=-2.3, max=4.6)`

## Analyzing Training Logs

If inference reveals problems, go back and check training logs:

```bash
# Check if MFA alignments were used
grep "MFA" training.log | head -20

# Check mel statistics during training
grep "mel_loss" training.log | tail -50

# Check for NaN gradients
grep "NaN\|nan" training.log
```

**Red flags in training logs:**
- "Phoneme count mismatch" errors → alignment issues
- Frequent NaN gradient skips → numerical instability
- Mel loss stuck at constant value → not learning
- Duration loss not decreasing → duration predictor not learning

## Next Steps Based on Findings

### If mel spectrogram looks correct but audio is garbage:
→ **Vocoder problem**, not model problem
- Try different vocoder checkpoint
- Check if vocoder expects different mel range
- Try Griffin-Lim to isolate vocoder issues

### If mel spectrogram is wrong (constant/wrong range):
→ **Training problem**, need to retrain
- Fix alignment usage (should be 100% MFA)
- Fix mel clamping in dataset
- Check loss weights and learning rate
- Train longer or use larger model

### If phonemes are wrong:
→ **Phoneme processor mismatch**
- Ensure g2p_en (ARPA) at both training and inference
- Regenerate model with correct processor
- Check `phoneme_processor.pkl` file

## Advanced Debugging: Compare Training Sample

To see if model learned correctly, compare inference output to a training sample:

```python
# In Python console
import torch
from data.ljspeech_dataset import LJSpeechDataset

# Load training sample
dataset = LJSpeechDataset("LJSpeech-1.1", phoneme_processor, sample_rate=22050)
sample = dataset[0]

print("Training mel range:", sample['mel_spec'].min().item(), sample['mel_spec'].max().item())
print("Training mel shape:", sample['mel_spec'].shape)
print("Training durations:", sample['durations_frames'][:20])
```

Compare these values to inference debug output. They should be similar!

## Getting Help

If you've followed this guide and still have issues, report:
1. Full debug output from test_inference_debug.py
2. Training loss curves (final mel_loss, dur_loss)
3. Sample training log (last 100 lines)
4. Model config (model_config.json)
5. Whether MFA alignments were used (check_fallback_usage.py output)
