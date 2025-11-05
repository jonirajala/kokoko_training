# CRITICAL: Missing PostNet Architecture

## Problem Identified

The model uses **only a single linear layer** to project decoder outputs to mel spectrograms:

```python
# kokoro/model.py line 118
self.mel_projection_out = nn.Linear(hidden_dim, mel_dim)  # 512 → 80
```

This is **severely underpowered** for mel spectrogram generation!

## Why This Causes High Mel Loss

From our diagnosis:
- **Mel loss stuck at ~1.8** even after 2000 iterations of overfitting
- **Correlation: 0.67** - The model learns general structure but not fine details
- **Gradient imbalance**: Decoder gets tiny gradients compared to duration predictor

A single linear layer cannot capture:
1. **Temporal dependencies** between mel frames
2. **Frequency-domain smoothness** (adjacent mel bins should be correlated)
3. **Fine harmonic structure** (formants, pitch harmonics)

## Industry Standard: PostNet

All successful TTS models use a **PostNet** for mel refinement:

### Tacotron 2 PostNet
```python
# 5 convolutional layers
Conv1d(mel_dim, 512, kernel_size=5, padding=2) → BatchNorm → Tanh
Conv1d(512, 512, kernel_size=5, padding=2) → BatchNorm → Tanh
Conv1d(512, 512, kernel_size=5, padding=2) → BatchNorm → Tanh
Conv1d(512, 512, kernel_size=5, padding=2) → BatchNorm → Tanh
Conv1d(512, mel_dim, kernel_size=5, padding=2) → BatchNorm
```

The PostNet is added as a **residual refinement**:
```python
mel_coarse = linear_projection(decoder_out)
mel_residual = postnet(mel_coarse)
mel_final = mel_coarse + mel_residual
```

### FastSpeech PostNet
Similar but lighter:
```python
# 3-5 convolutional layers
Conv1d(hidden_dim, hidden_dim, kernel_size=5) → LayerNorm → ReLU → Dropout
Conv1d(hidden_dim, hidden_dim, kernel_size=5) → LayerNorm → ReLU → Dropout
Conv1d(hidden_dim, mel_dim, kernel_size=5)
```

## Evidence from Our Tests

### Overfit Test Results
- **Duration loss**: 12.14 → 0.05 (excellent convergence!)
- **Mel loss**: 3.50 → 1.81 (stuck at high floor!)

The model CAN learn (duration predictor works), but the mel prediction path is **architecturally limited**.

### Gradient Analysis
```
Mel Projection Out gradient norm: 0.294
Decoder Layer 5 cross-attn: 0.000000 (VANISHED!)
```

The single linear layer gets gradients, but they can't flow back effectively through the decoder because the layer is too simple to provide meaningful learning signal.

### Correlation Analysis
```
Correlation between predicted and target: 0.6655
```

0.67 correlation means:
- ✅ Model learns **coarse structure** (energy envelope, rough spectral shape)
- ❌ Model fails at **fine structure** (formants, harmonics, transitions)

This is EXACTLY what you'd expect from a linear projection without temporal/frequency convolutions!

## The Fix

### Option 1: Add Full PostNet (Recommended)

Create a new `Postnet` module:

```python
class Postnet(nn.Module):
    """
    PostNet: 5 1-D convolution layers for mel spectrogram refinement
    """
    def __init__(self, mel_dim=80, postnet_dim=512, n_layers=5, kernel_size=5, dropout=0.5):
        super().__init__()

        self.convolutions = nn.ModuleList()

        # First layer: mel_dim → postnet_dim
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(mel_dim, postnet_dim, kernel_size=kernel_size,
                         stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.BatchNorm1d(postnet_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )

        # Middle layers: postnet_dim → postnet_dim
        for _ in range(n_layers - 2):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_dim, postnet_dim, kernel_size=kernel_size,
                             stride=1, padding=(kernel_size - 1) // 2, bias=False),
                    nn.BatchNorm1d(postnet_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )

        # Last layer: postnet_dim → mel_dim (no activation)
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_dim, mel_dim, kernel_size=kernel_size,
                         stride=1, padding=(kernel_size - 1) // 2, bias=False),
                nn.BatchNorm1d(mel_dim),
                nn.Dropout(dropout)
            )
        )

    def forward(self, x):
        # x: (batch, frames, mel_dim)
        # Conv1d expects: (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, mel_dim, frames)

        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)  # (batch, frames, mel_dim)
        return x
```

Update `KokoroModel.__init__`:
```python
# Replace single linear layer
self.mel_projection_coarse = nn.Linear(hidden_dim, mel_dim)
self.postnet = Postnet(mel_dim=mel_dim, postnet_dim=512, n_layers=5)
```

Update forward pass:
```python
# Coarse prediction
mel_coarse = self.mel_projection_coarse(decoder_outputs)

# Refine with postnet
mel_residual = self.postnet(mel_coarse)
predicted_mel_frames = mel_coarse + mel_residual
```

### Option 2: Lightweight PostNet (Faster Training)

Just 3 layers with smaller hidden dim:

```python
self.postnet = Postnet(mel_dim=80, postnet_dim=256, n_layers=3, kernel_size=5)
```

### Option 3: Multi-Layer Projection (Minimal Change)

If you want to avoid convolutions, at least add depth:

```python
self.mel_projection_out = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_dim // 2, mel_dim)
)
```

But this is still much weaker than a proper PostNet with convolutions!

## Expected Impact

With PostNet added:
- **Mel loss should drop to < 0.5** in real training (currently stuck at 0.4-1.8)
- **Overfit test should achieve < 0.1** mel loss
- **Audio quality should improve dramatically** (clearer speech, better prosody)

## Why This Wasn't Caught Earlier

1. **Low loss values misleading**: Mel loss of 0.4 seems good, but it's actually a high floor
2. **Duration predictor learning**: This masked the mel prediction problem
3. **No PostNet in original architecture**: Simplified model that skipped this critical component

## Related Issues

1. **Loss weight imbalance**: `duration_loss_weight=1.0` (should be 0.25)
   - This exacerbated the problem by giving even less gradient to mel prediction
2. **Gradient vanishing in decoder**: Single linear layer provides weak learning signal
3. **Missing intermediate supervision**: Could add pre-postnet loss for coarse prediction

## Action Items

1. ✅ **Identify root cause**: Missing PostNet architecture
2. ⏳ **Implement PostNet**: Add convolutional refinement network
3. ⏳ **Test on overfit**: Should achieve mel_loss < 0.1
4. ⏳ **Retrain from scratch**: Previous 100-epoch training is wasted due to architecture bug
5. ⏳ **Fix loss weights**: Set `duration_loss_weight=0.25` in config

## References

- Tacotron 2 paper: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"
- FastSpeech paper: "Fast Speech: Fast, Robust and Controllable Text to Speech"
- Both use PostNet for mel refinement!
