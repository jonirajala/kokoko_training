# Quick Start: Fixed Training Checklist

## ✅ What Was Fixed

1. **Added PostNet architecture** (kokoro/postnet.py)
   - 5-layer convolutional network for mel refinement
   - Residual connection: `mel_final = mel_coarse + postnet(mel_coarse)`

2. **Updated model** (kokoro/model.py)
   - Replaced single linear layer with coarse + PostNet
   - Updated both training and inference paths

3. **Fixed loss weights** (training/config_english.py)
   - Changed `duration_loss_weight` from 1.0 → 0.25
   - Fixes gradient imbalance (duration was drowning mel gradients)

4. **Updated test scripts**
   - test_overfit.py now monitors PostNet gradients
   - Uses correct loss weights

## 🧪 Before Starting Real Training

### Test 1: Verify Overfit Capability

Run the overfit test to confirm the model can learn:

```bash
python test_overfit.py
```

**Expected output:**
- Mel loss should drop below 0.5 (target < 0.3)
- Duration loss should drop below 0.1
- Training should NOT diverge or get stuck
- Final audio should be intelligible

**If it fails:**
- Check the TROUBLESHOOTING section in FIX_SUMMARY_ACTION_PLAN.md

### Test 2: Quick Inference Test

Verify the model architecture works for inference:

```bash
# This will fail with "no checkpoint" but should load the architecture
python inference_english.py --model ./nonexistent --text "test" 2>&1 | head -20
```

Should see model loading (even if checkpoint fails), confirming PostNet is integrated.

## 🚀 Start Training

### Option 1: From Scratch (Recommended)

```bash
# Backup old model if it exists
if [ -d "kokoro_english_model" ]; then
    mv kokoro_english_model kokoro_english_model_old_$(date +%Y%m%d)
fi

# Start fresh training
python training_english.py --batch_size 32 --epochs 100
```

### Option 2: Continue Training (NOT Recommended)

⚠️ **WARNING**: You CANNOT resume from old checkpoints because:
- Architecture changed (added PostNet layers)
- Loss weights changed (would cause gradient issues)
- Old weights are incompatible

If you try to resume, you'll get:
```
Error loading checkpoint: missing keys: ['postnet.convolutions...']
```

## 📊 Monitor Training Progress

### What to Watch

**First 10 epochs:**
```
Epoch 1:  Mel: 2.5, Dur: 8.0   ← High initial losses (normal)
Epoch 5:  Mel: 1.2, Dur: 1.0   ← Should be decreasing
Epoch 10: Mel: 0.8, Dur: 0.4   ← Target: mel < 1.0
```

**If mel loss > 1.5 after epoch 5:** Something is wrong! Check:
- Loss weights in config
- PostNet is actually being used
- No gradient issues in logs

### Early Quality Check

Test inference every 10 epochs:

```bash
# After epoch 10
python inference_english.py \
    --model ./kokoro_english_model \
    --text "The quick brown fox jumps over the lazy dog" \
    --output test_e10.wav \
    --debug

# Listen to test_e10.wav - should be barely intelligible but not pure noise
```

**Expected progression:**
- Epoch 10: Robotic, barely intelligible
- Epoch 20: Robotic but words are clear
- Epoch 50: Natural-sounding with minor artifacts
- Epoch 100: High quality, natural prosody

## 🐛 Common Issues

### Issue: "ModuleNotFoundError: No module named 'postnet'"

**Fix:**
```bash
# Verify file exists
ls kokoro/postnet.py

# If missing, the file wasn't saved properly
# Re-run the fix or copy from backup
```

### Issue: "AttributeError: 'KokoroModel' object has no attribute 'mel_projection_coarse'"

**Fix:**
```bash
# You're loading an old checkpoint with new code
# Must train from scratch - old checkpoints are incompatible
rm -rf kokoro_english_model/*.pth
python training_english.py
```

### Issue: Mel loss stuck > 1.0 after 20 epochs

**Diagnosis:**
```bash
# Check loss weight
grep duration_loss_weight training/config_english.py
# Should output: 0.25

# Check PostNet is loaded
python -c "from kokoro.model import KokoroModel; m = KokoroModel(96); print(hasattr(m, 'postnet'))"
# Should output: True
```

### Issue: Training crashes with CUDA OOM

**Fix:**
```bash
# PostNet adds 4.3M parameters, might need smaller batch size
python training_english.py --batch_size 16  # Reduce from 32
```

Or enable gradient checkpointing (already default):
```python
# In model __init__, verify:
gradient_checkpointing=True  # Should already be True
```

## 📈 Success Criteria

Training is working correctly if:

- [ ] Overfit test passes (mel < 0.5)
- [ ] Epoch 10: mel loss < 1.0
- [ ] Epoch 20: mel loss < 0.5, audio is intelligible
- [ ] Epoch 50: mel loss < 0.3, audio sounds natural
- [ ] No NaN losses or gradient explosions
- [ ] GPU memory usage stable (no OOM)

## 🎯 Final Checks Before Long Training

```bash
# 1. Verify PostNet exists
ls -lh kokoro/postnet.py

# 2. Check loss weight is correct
grep -n "duration_loss_weight.*0.25" training/config_english.py

# 3. Run overfit test
python test_overfit.py | tee overfit_test_results.log

# 4. Check overfit results
tail -50 overfit_test_results.log | grep "Mel Loss"
# Should see values < 0.5

# 5. Listen to overfit audio
# Open: overfit_test_output/overfit_test_output.wav
# Should be intelligible speech matching: "in being comparatively modern."
```

If ALL checks pass → Start training!
If ANY check fails → Debug before starting long training!

## ⏱️ Estimated Training Time

**On GPU (CUDA):**
- Epoch time: ~10-30 minutes (depends on GPU)
- 100 epochs: ~17-50 hours
- Checkpoints saved every 10 epochs

**On MPS (Mac):**
- Epoch time: ~30-60 minutes
- 100 epochs: ~50-100 hours

**On CPU:**
- Not recommended (too slow)
- Epoch time: ~4-8 hours
- Consider using Colab/AWS GPU instead

## 🔄 After Training Completes

1. **Test final model:**
   ```bash
   python inference_english.py \
       --model ./kokoro_english_model \
       --interactive
   ```

2. **Generate test samples:**
   ```bash
   python inference_english.py \
       --model ./kokoro_english_model \
       --text "This is a test of the text to speech system." \
       --output final_test.wav
   ```

3. **Compare to target quality:**
   - Listen to LJSpeech samples: `play LJSpeech-1.1/wavs/LJ001-0001.wav`
   - Listen to your model: `play final_test.wav`
   - Quality should be comparable (slightly more robotic is OK)

---

**Ready to start?** Run the overfit test first, then begin training if it passes!

```bash
# Quick command sequence:
python test_overfit.py && \
python training_english.py --batch_size 32 --epochs 100
```
