# Overfit Test Findings

## Problem Statement
After training for 100 epochs with mel loss reaching 0.40, the generated audio is garbage. Created an overfit test to verify if the model architecture can learn by training on a single sample.

## Key Findings

### 1. Mel Loss Not Learning in Overfit Test
**Symptom**: Mel loss stuck at ~1.98 for 800+ iterations, while duration loss decreases normally (0.066 → 0.035)

**Evidence**:
```
Iteration 600:
  Mel Loss: 1.982020 (STUCK!)
  Duration Loss: 0.066156 (learning)
  Mel projection gradient norm: 0.007390 (TINY!)
  Duration predictor gradient norm: 0.012948 (normal)
```

**Critical observation**: User reports "**in real training the loss goes down**" - meaning mel loss DOES decrease in full training but NOT in overfit test!

### 2. Differences Between Overfit Test and Real Training

#### Initially Missing (Now Fixed):
1. ✅ **Mixed precision** - Added BF16/FP16 autocast to match real training
2. ✅ **GradScaler** - Added for FP16 path (BF16 doesn't need it)
3. ✅ **Loss computation** - Matched exactly (including FP32 for BCE loss)
4. ✅ **Optimizer settings** - Using AdamW with lr=7e-5, weight_decay=0.01
5. ✅ **Gradient clipping** - Using clip_grad_norm with max_norm=1.0
6. ✅ **zero_grad(set_to_none=True)** - Matches real training

#### Potentially Problematic:
1. ❌ **Custom duration predictor initialization** - We added bias=1.79 and weight scaling, real training doesn't have this!
   - **REMOVED** - Now using default initialization
2. ❓ **Batch size** - Overfit uses batch_size=1, real training uses batch_size=32
   - LayerNorm should be fine with batch_size=1
   - But gradient statistics might differ
3. ❓ **Gradient checkpointing** - Disabled in overfit test for debugging
   - Real training uses it by default
   - Could affect gradient flow somehow?

### 3. Potential Root Causes

#### Theory 1: Batch Size Effects
- Real training: batch_size=32 → gradients averaged over 32 samples
- Overfit test: batch_size=1 → noisy gradients from single sample
- **Impact**: With batch_size=1, gradient variance is much higher
- **Solution**: Try increasing batch_size by repeating the same sample

#### Theory 2: Custom Initialization Breaking Learning
- We modified duration predictor initialization (bias=1.79, weight*=0.01)
- This might have broken some delicate balance in the model
- **Status**: FIXED - removed custom initialization

#### Theory 3: Learning Rate Too Low for Single Sample
- lr=7e-5 is optimized for batch_size=32
- For batch_size=1, might need higher LR (e.g., 7e-5 * sqrt(32) ≈ 4e-4)
- **Test**: Try running with higher LR

#### Theory 4: Weight Decay Interfering
- weight_decay=0.01 might prevent overfitting on single sample
- Purpose of overfit test is TO overfit!
- **Test**: Try weight_decay=0.0

#### Theory 5: Teacher Forcing Issue
- Model receives ground truth mel frames as input during training
- If decoder is not properly learning to transform encoder outputs
- **Evidence**: Mel gradient norm very small (0.007) suggests weak learning signal

### 4. Next Steps

1. **Run updated overfit test** (with removed custom init)
2. **If still not learning mel**:
   - Increase learning rate (try 5e-4)
   - Remove weight decay (set to 0.0)
   - Try batch_size > 1 by duplicating the sample
3. **Add more gradient debugging**:
   - Check gradients on encoder layers
   - Check gradients on mel_projection_in and mel_projection_out
   - Verify gradient flow through entire decoder
4. **Compare gradient statistics**:
   - Run 1 step of real training and log all gradient norms
   - Run 1 step of overfit test and compare
   - Find where gradients differ

### 5. Questions to Answer

1. Why does mel loss decrease in real training but not in overfit test?
2. Is there something about batch_size=1 that breaks learning?
3. Are gradients flowing correctly through the decoder in overfit test?
4. Is the learning rate appropriate for batch_size=1?

### 6. Current Test Configuration

```python
# Model
vocab_size=phoneme_processor.get_vocab_size()
mel_dim=80
hidden_dim=512
n_encoder_layers=6
n_heads=8
encoder_ff_dim=2048
n_decoder_layers=6
decoder_ff_dim=2048
gradient_checkpointing=False (disabled for debugging)

# Optimizer
optimizer = AdamW(lr=7e-5, weight_decay=0.01)

# Mixed Precision
BF16 on supported CUDA GPUs
FP16 with GradScaler on older GPUs
FP32 on CPU/MPS

# Loss Weights
duration_loss_weight = 1.0
stop_token_loss_weight = 0.1
```

## Conclusion

The overfit test reveals a fundamental issue: **the mel prediction network is not learning**, even though duration prediction learns fine. This suggests either:
1. A gradient flow issue specific to the decoder/mel prediction path
2. A learning rate / optimization issue specific to batch_size=1
3. An initialization problem that we introduced

The fact that real training (with batch_size=32) DOES see mel loss decrease suggests the issue is related to batch size or our test setup modifications.
