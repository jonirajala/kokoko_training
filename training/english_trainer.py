#!/usr/bin/env python3
"""
English TTS Trainer - extends KokoroTrainer for English dataset
"""

from .trainer import KokoroTrainer
from data.ljspeech_dataset import LJSpeechDataset, collate_fn, LengthBasedBatchSampler
from data.english_phoneme_processor import EnglishPhonemeProcessor
from torch.utils.data import DataLoader
from .device_type import DeviceType
import logging

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnglishTrainer(KokoroTrainer):
    """
    Trainer for English TTS using LJSpeech dataset.

    Extends KokoroTrainer but uses English dataset and phoneme processor.
    """

    def __init__(self, config):
        """Initialize English trainer with LJSpeech dataset"""

        # We need to initialize the parent class partially, then override the dataset
        # Call parent __init__ but we'll override dataset creation

        # First, manually set up what parent __init__ does before creating dataset
        self.config = config
        self.device = config.device if hasattr(config, 'device') else 'cpu'
        if isinstance(self.device, str):
            import torch
            self.device = torch.device(self.device)

        # Initialize memory manager and other components
        from .adaptive_memory_manager import AdaptiveMemoryManager
        self.memory_manager = AdaptiveMemoryManager(self.device, config)

        # Mixed precision setup with automatic BF16/FP16 detection
        import torch
        self.use_mixed_precision = getattr(config, 'use_mixed_precision', True)

        if self.use_mixed_precision and self.device.type == DeviceType.CUDA.value:
            # Auto-detect best dtype for CUDA devices
            if torch.cuda.is_bf16_supported():
                # ✅ Prefer BF16 on modern GPUs (Ampere/Ada/Hopper)
                self.autocast_dtype = torch.bfloat16
                self.mixed_precision_dtype = torch.bfloat16
                self.scaler = None  # No GradScaler needed for BF16
                self.use_grad_scaler = False
                self.device_type = 'cuda'
                logger.info("✓ Using bfloat16 autocast on CUDA (no GradScaler needed)")
                logger.info("  GPU supports BF16 - optimal stability without scaling")
            else:
                # Fallback to FP16 with conservative GradScaler for older GPUs
                self.autocast_dtype = torch.float16
                self.mixed_precision_dtype = torch.float16
                self.scaler = torch.cuda.amp.GradScaler(
                    init_scale=2**12,  # Conservative initial scale (4096)
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=1000,
                    enabled=True
                )
                self.use_grad_scaler = True
                self.max_grad_scale = 2**15  # Maximum scale limit (32768)
                self.device_type = 'cuda'
                logger.info("✓ Using float16 autocast on CUDA with GradScaler fallback")
                logger.info("  GPU does not support BF16 - using FP16 with conservative scaling")

        elif self.use_mixed_precision and self.device.type == DeviceType.MPS.value:
            # MPS (Apple Silicon) handling
            config_dtype = getattr(config, 'mixed_precision_dtype', torch.float16)
            if config_dtype == torch.bfloat16:
                self.autocast_dtype = torch.bfloat16
                self.mixed_precision_dtype = torch.bfloat16
                self.scaler = None
                self.use_grad_scaler = False
                logger.info("✓ Using bfloat16 autocast on MPS (no scaler needed)")
            else:
                from .mps_grad_scaler import MPSGradScaler
                self.autocast_dtype = torch.float16
                self.mixed_precision_dtype = torch.float16
                self.scaler = MPSGradScaler(
                    init_scale=2**12,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=1000
                )
                self.use_grad_scaler = True
                logger.info("✓ Using float16 autocast on MPS with custom scaler")
            self.device_type = DeviceType.MPS.value

        else:
            # CPU or mixed precision disabled
            self.use_mixed_precision = False
            self.scaler = None
            self.use_grad_scaler = False
            self.autocast_dtype = torch.float32
            self.mixed_precision_dtype = torch.float32
            self.device_type = self.device.type
            if self.device.type == DeviceType.CUDA.value or self.device.type == DeviceType.MPS.value:
                logger.info("Mixed precision training disabled by configuration")
            else:
                logger.info(f"Mixed precision not supported on {self.device.type}, using FP32")

        # NOW create our English dataset
        logger.info("Loading English LJSpeech dataset...")
        self.dataset = LJSpeechDataset(config.data_dir, config)
        logger.info(f"Loaded {len(self.dataset)} samples")

        # Create batch sampler
        self.batch_sampler = LengthBasedBatchSampler(
            dataset=self.dataset,
            batch_size=config.batch_size,
            drop_last=True,
            shuffle=True
        )

        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=collate_fn,
            num_workers=getattr(config, 'num_workers', 2),
            pin_memory=getattr(config, 'pin_memory', False) and self.device.type == DeviceType.CUDA.value,
            prefetch_factor=3 if getattr(config, 'num_workers', 2) > 0 else None,
            persistent_workers=getattr(config, 'num_workers', 2) > 0
        )

        # Initialize model with English vocab size
        from kokoro.model import KokoroModel
        vocab_size = self.dataset.phoneme_processor.get_vocab_size()
        logger.info(f"English vocabulary size: {vocab_size}")

        self.model = KokoroModel(
            vocab_size=vocab_size,
            mel_dim=config.n_mels,
            hidden_dim=config.hidden_dim,
            n_encoder_layers=getattr(config, 'n_encoder_layers', 6),
            n_decoder_layers=getattr(config, 'n_decoder_layers', 6),
            n_heads=getattr(config, 'n_heads', 8),
            encoder_ff_dim=getattr(config, 'encoder_ff_dim', 2048),
            encoder_dropout=getattr(config, 'encoder_dropout', 0.1),
            decoder_ff_dim=getattr(config, 'decoder_ff_dim', 2048),
            max_decoder_seq_len=getattr(config, 'max_decoder_seq_len', 4000),
            enable_profiling=getattr(config, 'enable_profiling', False),
            gradient_checkpointing=getattr(config, 'gradient_checkpointing', True),
            checkpoint_segments=getattr(config, 'checkpoint_segments', 2)
        )
        self.model.to(self.device)

        # Log model info
        model_info = self.model.get_model_info()
        logger.info(f"Model initialized with {model_info['total_parameters']:,} parameters ({model_info['model_size_mb']:.1f} MB)")

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, 'weight_decay', 0.01),
            eps=getattr(config, 'adam_eps', 1e-8),
            betas=getattr(config, 'adam_betas', (0.9, 0.999))
        )

        # Loss functions
        import torch.nn as nn
        self.criterion_mel = nn.L1Loss(reduction='none')
        self.criterion_duration = nn.MSELoss(reduction='none')
        self.criterion_stop_token = nn.BCEWithLogitsLoss(reduction='none')

        # Learning rate scheduler
        # Convert T_0 from epochs to batches since we call scheduler.step() per batch
        num_batches_per_epoch = len(self.dataloader)
        T_0_epochs = getattr(config, 'lr_T_0', 20)
        T_0_batches = T_0_epochs * num_batches_per_epoch

        logger.info(f"Learning rate scheduler: CosineAnnealingWarmRestarts")
        logger.info(f"  T_0: {T_0_epochs} epochs = {T_0_batches} batches")
        logger.info(f"  T_mult: {getattr(config, 'lr_T_mult', 2)}")
        logger.info(f"  eta_min: {getattr(config, 'lr_eta_min', 1e-6)}")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0_batches,
            T_mult=getattr(config, 'lr_T_mult', 2),
            eta_min=getattr(config, 'lr_eta_min', 1e-6)
        )

        # Training state
        self.start_epoch = 0
        self.best_loss = float('inf')

        # Stats
        self.mixed_precision_stats = {
            'scale_updates': 0,
            'scale_decreases': 0,
            'overflow_count': 0,
            'successful_steps': 0,
            'skipped_steps': 0
        }

        # Profiling
        self.profiler = None
        self.profiling_stats = {}
        self.memory_snapshots = []

        import os
        import datetime
        self.log_dir = os.path.join(config.output_dir, "profiler_logs",
                                    datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)

        # Interbatch profiler
        from .interbatch_profiler import InterbatchProfiler
        self.interbatch_profiler = InterbatchProfiler(config)

        # Memory management
        self.enable_adaptive_memory = getattr(config, 'enable_adaptive_memory', True)
        self.memory_report_interval = getattr(config, 'memory_report_interval', 500)

        # W&B initialization
        self.use_wandb = getattr(config, 'use_wandb', False) and WANDB_AVAILABLE
        self.wandb_run = None

        logger.info(f"W&B requested: {getattr(config, 'use_wandb', False)}, W&B available: {WANDB_AVAILABLE}")

        if self.use_wandb:
            logger.info("Initializing W&B logging...")
            self._init_wandb()
        elif getattr(config, 'use_wandb', False) and not WANDB_AVAILABLE:
            logger.warning("W&B logging requested but wandb not installed. Install with: pip install wandb")
        else:
            logger.info("W&B logging disabled")

        logger.info("English trainer initialized successfully")

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        if not WANDB_AVAILABLE:
            return

        try:
            # Prepare wandb config
            wandb_config = {
                # Model architecture
                "model": "Kokoro-English-TTS",
                "vocab_size": len(self.dataset.phoneme_processor.phoneme_to_id),
                "hidden_dim": self.config.hidden_dim,
                "n_encoder_layers": getattr(self.config, 'n_encoder_layers', 6),
                "n_decoder_layers": getattr(self.config, 'n_decoder_layers', 6),
                "n_heads": getattr(self.config, 'n_heads', 8),

                # Training params
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "device": str(self.device),
                "mixed_precision": self.use_mixed_precision,
                "gradient_checkpointing": getattr(self.config, 'gradient_checkpointing', True),

                # Dataset
                "dataset": "LJSpeech",
                "dataset_size": len(self.dataset),
                "sample_rate": self.config.sample_rate,

                # Loss weights
                "duration_loss_weight": self.config.duration_loss_weight,
                "stop_token_loss_weight": self.config.stop_token_loss_weight,
            }

            # Get model parameter count
            model_info = self.model.get_model_info()
            wandb_config["total_parameters"] = model_info['total_parameters']
            wandb_config["model_size_mb"] = model_info['model_size_mb']

            # Initialize wandb
            self.wandb_run = wandb.init(
                project=getattr(self.config, 'wandb_project', 'kokoro-english-tts'),
                entity=getattr(self.config, 'wandb_entity', None),
                name=getattr(self.config, 'wandb_run_name', None),
                tags=getattr(self.config, 'wandb_tags', None),
                notes=getattr(self.config, 'wandb_notes', None),
                config=wandb_config,
                resume="allow"  # Allow resuming if run exists
            )

            # Watch model
            wandb.watch(self.model, log="all", log_freq=100)

            logger.info(f"W&B initialized: {self.wandb_run.url}")

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
            self.wandb_run = None

    def log_to_wandb(self, metrics: dict, step: int = None, commit: bool = True):
        """Log metrics to Weights & Biases"""
        if not self.use_wandb or not self.wandb_run:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def train_epoch(self, epoch: int):
        """Override train_epoch to add per-batch W&B logging"""
        from tqdm import tqdm
        import torch
        import torch.profiler

        self.model.train()
        total_loss_epoch = 0.0
        mel_loss_epoch = 0.0
        dur_loss_epoch = 0.0
        stop_loss_epoch = 0.0

        num_batches = len(self.dataloader)

        # Calculate base global step for this epoch
        base_global_step = epoch * num_batches

        # Use parent's profiling logic if needed
        is_profiling_epoch = (epoch == self.config.profile_epoch_start) and self.config.enable_profiling
        enable_interbatch_profiling = getattr(self.config, 'enable_interbatch_profiling', False)

        if is_profiling_epoch:
            logger.info(f"Starting profiler for epoch {epoch+1}")
            self.reset_profiling_stats()
            self.profiler = self.start_torch_profiler()
            self.profiler.__enter__()

        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            global_step = base_global_step + batch_idx

            try:
                # Adaptive memory cleanup
                cleanup_result = self.adaptive_memory_cleanup(batch_idx)

                # Move data to device
                with torch.profiler.record_function("Data_Loading"):
                    non_blocking = self.device.type == 'cuda'
                    mel_specs = batch['mel_specs'].to(self.device, non_blocking=non_blocking)
                    phoneme_indices = batch['phoneme_indices'].to(self.device, non_blocking=non_blocking)
                    phoneme_durations = batch['phoneme_durations'].to(self.device, non_blocking=non_blocking)
                    stop_token_targets = batch['stop_token_targets'].to(self.device, non_blocking=non_blocking)
                    mel_lengths = batch['mel_lengths'].to(self.device, non_blocking=non_blocking)
                    phoneme_lengths = batch['phoneme_lengths'].to(self.device, non_blocking=non_blocking)

                self.optimizer.zero_grad()

                # Forward pass
                with torch.profiler.record_function("Model_Forward"):
                    if self.use_mixed_precision:
                        with self.get_autocast_context():
                            predicted_mel, predicted_log_durations, predicted_stop_logits = \
                                self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)
                    else:
                        predicted_mel, predicted_log_durations, predicted_stop_logits = \
                            self.model(phoneme_indices, mel_specs, phoneme_durations, stop_token_targets)

                # Loss calculation
                with torch.profiler.record_function("Loss_Calculation"):
                    if self.use_mixed_precision:
                        with self.get_autocast_context():
                            total_loss, loss_mel, loss_duration, loss_stop_token = self._calculate_losses(
                                predicted_mel, predicted_log_durations, predicted_stop_logits,
                                mel_specs, phoneme_durations, stop_token_targets,
                                mel_lengths, phoneme_lengths
                            )
                    else:
                        total_loss, loss_mel, loss_duration, loss_stop_token = self._calculate_losses(
                            predicted_mel, predicted_log_durations, predicted_stop_logits,
                            mel_specs, phoneme_durations, stop_token_targets,
                            mel_lengths, phoneme_lengths
                        )

                # Backward pass
                # ========== Backward + Optimizer Step (Simplified) ==========
                with torch.profiler.record_function("Backward_Pass"):
                    if self.use_mixed_precision and self.autocast_dtype == torch.bfloat16:
                        # ✅ BF16 path (no GradScaler needed - inherently stable)
                        self.optimizer.zero_grad(set_to_none=True)
                        total_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                        if not torch.isfinite(grad_norm):
                            logger.warning(f"[Batch {batch_idx}] Non-finite grad norm ({grad_norm:.2f}). Skipping batch.")
                            self.optimizer.zero_grad(set_to_none=True)
                            continue

                        self.optimizer.step()

                    elif self.use_mixed_precision and self.use_grad_scaler:
                        # FP16 path with GradScaler (backward compatibility)
                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                        if not torch.isfinite(grad_norm):
                            logger.warning(f"[Batch {batch_idx}] Non-finite grad norm ({grad_norm:.2f}). Skipping batch.")
                            self.optimizer.zero_grad(set_to_none=True)
                            self.scaler.update()
                            continue

                        self.scaler.step(self.optimizer)
                        old_scale = self.scaler.get_scale()
                        self.scaler.update()
                        new_scale = self.scaler.get_scale()

                        # Cap grad scale if needed (prevent unbounded growth)
                        if hasattr(self, 'max_grad_scale') and self.max_grad_scale is not None and new_scale > self.max_grad_scale:
                            try:
                                self.scaler._scale.fill_(self.max_grad_scale)
                                if batch_idx % 500 == 0:
                                    logger.info(f"Grad scale capped at {self.max_grad_scale} (was {float(new_scale):.0f})")
                            except Exception:
                                logger.warning("GradScaler._scale cap failed (internal API change)")

                    else:
                        # FP32 fallback (no mixed precision)
                        self.optimizer.zero_grad(set_to_none=True)
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()

                # Cache loss values (single .item() call per loss - no duplicate GPU syncs)
                loss_total_val = total_loss.item()
                loss_mel_val = loss_mel.item()
                loss_dur_val = loss_duration.item()
                loss_stop_val = loss_stop_token.item()

                # Accumulate losses using cached values
                total_loss_epoch += loss_total_val
                mel_loss_epoch += loss_mel_val
                dur_loss_epoch += loss_dur_val
                stop_loss_epoch += loss_stop_val

                # W&B logging per batch (every 50 batches to reduce queue pressure)
                # Use commit=False to prevent blocking on queue full
                if self.use_wandb and batch_idx % 50 == 0:
                    wandb_metrics = {
                        "train/batch_total_loss": loss_total_val,
                        "train/batch_mel_loss": loss_mel_val,
                        "train/batch_duration_loss": loss_dur_val,
                        "train/batch_stop_loss": loss_stop_val,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch + 1,
                    }

                    # Only log grad_scale if using FP16 with GradScaler
                    if self.use_mixed_precision and self.use_grad_scaler and self.scaler:
                        wandb_metrics["train/grad_scale"] = self.scaler.get_scale()

                    # commit=False prevents blocking on network I/O
                    self.log_to_wandb(wandb_metrics, step=global_step, commit=False)

                # Update progress bar using cached values
                postfix_dict = {
                    'total_loss': loss_total_val,
                    'mel_loss': loss_mel_val,
                    'dur_loss': loss_dur_val,
                    'stop_loss': loss_stop_val,
                    'lr': self.optimizer.param_groups[0]['lr']
                }

                if self.use_mixed_precision:
                    if self.use_grad_scaler and self.scaler:
                        postfix_dict['scale'] = f"{self.scaler.get_scale():.0f}"
                    else:
                        postfix_dict['dtype'] = 'bf16'

                if self.enable_adaptive_memory:
                    postfix_dict['mem'] = cleanup_result.get('pressure_level', 'unknown')[:3]
                    if cleanup_result.get('cleaned', False):
                        postfix_dict['mem'] += '*'

                progress_bar.set_postfix(postfix_dict)

                # Step the learning rate scheduler after each batch
                self.scheduler.step()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM error at batch {batch_idx}: {e}")
                    can_continue = self.handle_oom_with_adaptive_cleanup(batch_idx, e)
                    if can_continue:
                        continue
                    else:
                        raise e
                else:
                    raise e

        # Calculate epoch averages
        avg_total_loss = total_loss_epoch / num_batches
        avg_mel_loss = mel_loss_epoch / num_batches
        avg_dur_loss = dur_loss_epoch / num_batches
        avg_stop_loss = stop_loss_epoch / num_batches

        # Log epoch summary to W&B
        if self.use_wandb:
            wandb_metrics = {
                "epoch": epoch + 1,
                "train/epoch_total_loss": avg_total_loss,
                "train/epoch_mel_loss": avg_mel_loss,
                "train/epoch_duration_loss": avg_dur_loss,
                "train/epoch_stop_loss": avg_stop_loss,
            }

            # Add memory stats if available
            if self.enable_adaptive_memory:
                memory_report = self.memory_manager.get_memory_report()
                wandb_metrics.update({
                    "memory/pressure": {
                        "low": 0, "moderate": 1, "high": 2, "critical": 3
                    }.get(memory_report['current_pressure'], 0),
                    "memory/cleanup_count": memory_report['cleanup_count'],
                    "memory/cleanup_overhead_percent": memory_report['cleanup_overhead_percent'],
                })

            # commit=True at epoch end to flush any pending logs
            self.log_to_wandb(wandb_metrics, step=base_global_step + num_batches, commit=True)

        # Cleanup profiler if it was started
        if self.profiler:
            self.profiler.__exit__(None, None, None)
            self.profiler = None

        return avg_total_loss, avg_mel_loss, avg_dur_loss, avg_stop_loss

    def train(self):
        """Override train to properly finish W&B run"""
        try:
            # Call parent's train method
            super().train()

            # Mark run as finished
            if self.use_wandb and self.wandb_run:
                wandb.finish()
                logger.info("W&B run finished successfully")

        except Exception as e:
            # Ensure W&B run is finished even on error
            if self.use_wandb and self.wandb_run:
                wandb.finish(exit_code=1)
            raise e

    def get_autocast_context(self):
        """Get the appropriate autocast context for the device"""
        import torch

        if not self.use_mixed_precision:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()

        if self.device_type == DeviceType.CUDA.value:
            # Use new API: torch.amp.autocast instead of torch.cuda.amp.autocast
            return torch.amp.autocast("cuda", dtype=torch.float16)
        elif self.device_type == DeviceType.MPS.value:
            # MPS autocast with proper torch.dtype
            return torch.amp.autocast("mps", dtype=torch.float16)
        else:
            from contextlib import nullcontext
            return nullcontext()
