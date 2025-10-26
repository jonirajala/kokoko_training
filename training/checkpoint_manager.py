#!/usr/bin/env python3
"""
Checkpoint management utilities
"""

import os
import torch
import pickle
from pathlib import Path
from typing import Optional, Tuple
import logging

from .config_english import EnglishTrainingConfig as TrainingConfig
from data.english_phoneme_processor import EnglishPhonemeProcessor

logger = logging.getLogger(__name__)


def save_phoneme_processor(processor: EnglishPhonemeProcessor, output_dir: str):
    """Save phoneme processor separately as pickle file"""
    processor_path = os.path.join(output_dir, "phoneme_processor.pkl")
    with open(processor_path, 'wb') as f:
        pickle.dump(processor.to_dict(), f)
    logger.info(f"Phoneme processor saved: {processor_path}")


def load_phoneme_processor(output_dir: str) -> EnglishPhonemeProcessor:
    """Load phoneme processor from pickle file"""
    processor_path = os.path.join(output_dir, "phoneme_processor.pkl")
    with open(processor_path, 'rb') as f:
        processor_data = pickle.load(f)
    processor = EnglishPhonemeProcessor.from_dict(processor_data)
    logger.info(f"Phoneme processor loaded: {processor_path}")
    return processor


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    loss: float,
    config: TrainingConfig,
    output_dir: str
):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config
    }
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    output_dir: str
) -> Tuple[int, float, EnglishPhonemeProcessor]:
    """Load checkpoint with robust error handling for optimizer/scheduler state"""
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Add safe globals for our custom classes (PyTorch 2.6+ only)
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([TrainingConfig, EnglishPhonemeProcessor])

    try:
        # Try loading with weights_only=True first (new default in PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    except Exception as e:
        logger.warning(f"Loading with weights_only=True failed: {e}")
        logger.info("Trying to load with weights_only=False for compatibility...")

        try:
            # Try loading with weights_only=False for older checkpoints
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            logger.error(f"Error loading checkpoint even with weights_only=False: {e2}")
            raise e2

    # At this point we have the checkpoint loaded
    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model state dict: {e}")
        raise

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']

    # Try to load optimizer state with error handling
    try:
        if 'optimizer_state_dict' in checkpoint:
            # The optimizer state dict might have issues, try to load it carefully
            optimizer_state = checkpoint['optimizer_state_dict']

            # Check if state dict is valid
            if optimizer_state and 'state' in optimizer_state:
                optimizer.load_state_dict(optimizer_state)
                logger.info("Optimizer state loaded successfully")
            else:
                logger.warning("Optimizer state dict is malformed, starting fresh optimizer")
        else:
            logger.warning("No optimizer state in checkpoint, starting fresh optimizer")
    except Exception as e:
        logger.warning(f"Failed to load optimizer state: {e}")
        logger.info("Continuing with fresh optimizer state (not a critical error)")

    # Try to load scheduler state with error handling
    try:
        if 'scheduler_state_dict' in checkpoint:
            scheduler_state = checkpoint['scheduler_state_dict']

            # Check if state dict is valid
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)
                logger.info("Scheduler state loaded successfully")
            else:
                logger.warning("Scheduler state dict is malformed, starting fresh scheduler")
        else:
            logger.warning("No scheduler state in checkpoint, starting fresh scheduler")
    except Exception as e:
        logger.warning(f"Failed to load scheduler state: {e}")
        logger.info("Continuing with fresh scheduler state (not a critical error)")

    # Load or create phoneme processor
    if 'phoneme_processor' in checkpoint:
        phoneme_processor = checkpoint['phoneme_processor']
        logger.info("Phoneme processor loaded from checkpoint")
    else:
        # Create fresh phoneme processor if not in checkpoint
        phoneme_processor = EnglishPhonemeProcessor()
        logger.info("Created fresh phoneme processor")

    logger.info(f"Resumed from epoch {start_epoch} with loss {best_loss:.4f}")
    return start_epoch, best_loss, phoneme_processor


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the output directory"""
    checkpoint_dir = Path(output_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoint_files:
        return None

    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest_checkpoint = checkpoint_files[-1]

    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return str(latest_checkpoint)


def save_final_model(model: torch.nn.Module, config: TrainingConfig, output_dir: str):
    """Save final model"""
    final_model_path = os.path.join(output_dir, "kokoro_russian_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_model_path)
    logger.info(f"Final model saved: {final_model_path}")
