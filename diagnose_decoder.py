#!/usr/bin/env python3
"""
Diagnose why the decoder can't learn to predict mel spectrograms properly.

This script checks:
1. Gradient flow through all decoder components
2. Activation statistics (dead neurons?)
3. Teacher forcing behavior
4. Output saturation
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the overfitted model and training sample
model_dir = Path("overfit_test_output")
checkpoint = torch.load(model_dir / "overfit_model.pth", map_location='cpu')
sample = torch.load(model_dir / "training_sample.pt", map_location='cpu')

# Load model
from kokoro.model import KokoroModel
from data.english_phoneme_processor import EnglishPhonemeProcessor
import pickle

with open(model_dir / "phoneme_processor.pkl", 'rb') as f:
    phoneme_processor_dict = pickle.load(f)

phoneme_processor = EnglishPhonemeProcessor()
phoneme_processor.from_dict(phoneme_processor_dict)

vocab_size = phoneme_processor.get_vocab_size()
model = KokoroModel(
    vocab_size=vocab_size,
    mel_dim=80,
    hidden_dim=512,
    n_encoder_layers=6,
    n_heads=8,
    encoder_ff_dim=2048,
    encoder_dropout=0.1,
    n_decoder_layers=6,
    decoder_ff_dim=2048,
    max_decoder_seq_len=4000,
    enable_profiling=False,
    gradient_checkpointing=False
)

model.load_state_dict(checkpoint['model_state_dict'])
model.train()  # Training mode

logger.info("="*70)
logger.info("DECODER GRADIENT FLOW DIAGNOSIS")
logger.info("="*70)

# Prepare inputs
phoneme_indices = torch.tensor(sample['phoneme_indices']).unsqueeze(0)
mel_spec = sample['mel_spec'].unsqueeze(0)
durations = torch.tensor(sample['phoneme_durations']).unsqueeze(0)
num_frames = mel_spec.shape[1]
stop_tokens = torch.zeros(1, num_frames)
stop_tokens[:, -1] = 1.0

logger.info(f"\nInput shapes:")
logger.info(f"  Phonemes: {phoneme_indices.shape}")
logger.info(f"  Mel spec: {mel_spec.shape}")
logger.info(f"  Durations: {durations.shape}")

# Forward pass with gradient tracking - USE TEACHER FORCING (same as overfit test)
mel_pred, dur_pred, stop_pred = model.forward_training(
    phoneme_indices=phoneme_indices,
    mel_specs=mel_spec,
    phoneme_durations=durations,
    stop_token_targets=stop_tokens,
    text_padding_mask=None,
    mel_padding_mask=None,
    use_gt_durations=True  # Bypass duration predictor, use ground truth
)

# Compute losses - matching overfit test configuration
mel_loss_weight = 5.0
duration_loss_weight = 0.0  # Zero - using ground truth durations
stop_loss_weight = 0.05

mel_loss = nn.L1Loss(reduction='mean')(mel_pred, mel_spec)
dur_loss = torch.zeros((), device=mel_spec.device)  # Not training duration predictor
stop_loss = nn.BCEWithLogitsLoss()(stop_pred.squeeze(-1), stop_tokens)
total_loss = mel_loss_weight * mel_loss + duration_loss_weight * dur_loss + stop_loss_weight * stop_loss

logger.info(f"\nLosses:")
logger.info(f"  Mel: {mel_loss.item():.4f}")
logger.info(f"  Duration: {dur_loss.item():.4f}")
logger.info(f"  Stop: {stop_loss.item():.4f}")
logger.info(f"  Total: {total_loss.item():.4f}")

# Backward pass
total_loss.backward()

logger.info(f"\n{'='*70}")
logger.info("GRADIENT ANALYSIS")
logger.info(f"{'='*70}")

# Check gradients for each component
components = {
    'Text Embedding': model.text_embedding.weight,
    'Encoder Layer 0 (attn)': model.transformer_encoder_layers[0].self_attn.w_q.weight,
    'Encoder Layer 5 (attn)': model.transformer_encoder_layers[5].self_attn.w_q.weight,
    'Duration Predictor (first)': model.duration_predictor[0].weight,
    'Duration Predictor (last)': model.duration_predictor[-1].weight,
    'Mel Projection In': model.mel_projection_in.weight,
    'Decoder Layer 0 (self-attn)': model.decoder.layers[0].self_attn.w_q.weight,
    'Decoder Layer 5 (self-attn)': model.decoder.layers[5].self_attn.w_q.weight,
    'Decoder Layer 0 (cross-attn)': model.decoder.layers[0].cross_attn.w_q.weight,
    'Decoder Layer 5 (cross-attn)': model.decoder.layers[5].cross_attn.w_q.weight,
    'Mel Projection Coarse': model.mel_projection_coarse.weight,
    'PostNet Layer 0': model.postnet.convolutions[0][0].weight,
    'PostNet Layer 4 (last)': model.postnet.convolutions[4][0].weight,
    'Stop Token Predictor': model.stop_token_predictor.weight,
}

max_component_name_len = max(len(name) for name in components.keys())

logger.info(f"\n{'Component':<{max_component_name_len}}  Grad Norm    Grad Mean     Grad Std      Weight Norm")
logger.info("-" * 100)

for name, param in components.items():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        weight_norm = param.norm().item()
        logger.info(f"{name:<{max_component_name_len}}  {grad_norm:10.6f}   {grad_mean:10.6f}   {grad_std:10.6f}   {weight_norm:10.4f}")
    else:
        logger.info(f"{name:<{max_component_name_len}}  NO GRADIENT!")

# Check for vanishing/exploding gradients
logger.info(f"\n{'='*70}")
logger.info("GRADIENT HEALTH CHECK")
logger.info(f"{'='*70}")

all_grads = []
for name, param in model.named_parameters():
    if param.grad is not None:
        all_grads.append(param.grad.norm().item())

if all_grads:
    min_grad = min(all_grads)
    max_grad = max(all_grads)
    mean_grad = sum(all_grads) / len(all_grads)

    logger.info(f"\nGradient statistics across all parameters:")
    logger.info(f"  Min gradient norm: {min_grad:.8f}")
    logger.info(f"  Max gradient norm: {max_grad:.6f}")
    logger.info(f"  Mean gradient norm: {mean_grad:.6f}")
    logger.info(f"  Gradient ratio (max/min): {max_grad/min_grad if min_grad > 0 else float('inf'):.2f}")

    if min_grad < 1e-6:
        logger.warning(f"  ⚠️  VANISHING GRADIENTS detected! Min grad = {min_grad:.8f}")
    if max_grad > 100:
        logger.warning(f"  ⚠️  EXPLODING GRADIENTS detected! Max grad = {max_grad:.2f}")

    # Check decoder specifically
    decoder_grads = []
    encoder_grads = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'decoder' in name or 'mel_projection' in name:
                decoder_grads.append(grad_norm)
            elif 'encoder' in name or 'text_embedding' in name:
                encoder_grads.append(grad_norm)

    if decoder_grads and encoder_grads:
        mean_decoder_grad = sum(decoder_grads) / len(decoder_grads)
        mean_encoder_grad = sum(encoder_grads) / len(encoder_grads)

        logger.info(f"\nDecoder vs Encoder gradient comparison:")
        logger.info(f"  Mean decoder gradient: {mean_decoder_grad:.6f}")
        logger.info(f"  Mean encoder gradient: {mean_encoder_grad:.6f}")
        logger.info(f"  Ratio (decoder/encoder): {mean_decoder_grad/mean_encoder_grad if mean_encoder_grad > 0 else float('inf'):.2f}")

        if mean_decoder_grad < mean_encoder_grad * 0.1:
            logger.warning(f"  ⚠️  DECODER GRADIENTS are significantly smaller than encoder!")
            logger.warning(f"     This suggests gradient flow issues through the decoder!")

# Check mel prediction quality
logger.info(f"\n{'='*70}")
logger.info("MEL PREDICTION ANALYSIS")
logger.info(f"{'='*70}")

mel_diff = (mel_pred - mel_spec).abs()
logger.info(f"\nMel prediction error distribution:")
logger.info(f"  Mean absolute error: {mel_diff.mean().item():.4f}")
logger.info(f"  Std of error: {mel_diff.std().item():.4f}")
logger.info(f"  Max error: {mel_diff.max().item():.4f}")
logger.info(f"  Min error: {mel_diff.min().item():.4f}")

logger.info(f"\nMel prediction statistics:")
logger.info(f"  Predicted range: [{mel_pred.min().item():.3f}, {mel_pred.max().item():.3f}]")
logger.info(f"  Target range: [{mel_spec.min().item():.3f}, {mel_spec.max().item():.3f}]")
logger.info(f"  Predicted mean: {mel_pred.mean().item():.3f}")
logger.info(f"  Target mean: {mel_spec.mean().item():.3f}")
logger.info(f"  Predicted std: {mel_pred.std().item():.3f}")
logger.info(f"  Target std: {mel_spec.std().item():.3f}")

# Check if prediction is correlated with target
correlation = torch.corrcoef(torch.stack([
    mel_pred.flatten(),
    mel_spec.flatten()
]))[0, 1]
logger.info(f"\nCorrelation between predicted and target: {correlation.item():.4f}")
if correlation < 0.5:
    logger.warning(f"  ⚠️  LOW CORRELATION! Model is not learning the target distribution well!")

logger.info(f"\n{'='*70}")
logger.info("DIAGNOSIS COMPLETE")
logger.info(f"{'='*70}")
