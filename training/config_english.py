#!/usr/bin/env python3
"""
Training Configuration for English LJSpeech Dataset
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnglishTrainingConfig:
    """Training configuration optimized for LJSpeech English dataset"""

    # Dataset paths
    data_dir: str = "LJSpeech-1.1"
    output_dir: str = "output_models_english"

    # Basic training parameters
    num_epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Learning rate scheduler (Cosine Annealing with Warm Restarts)
    lr_T_0: int = 20          # Number of epochs for first restart
    lr_T_mult: int = 2        # Factor to increase T_i after restart
    lr_eta_min: float = 1e-6  # Minimum learning rate

    # Optimizer parameters
    weight_decay: float = 0.01
    adam_eps: float = 1e-8
    adam_betas: tuple = (0.9, 0.999)

    # Model architecture parameters
    n_mels: int = 80                    # Number of mel frequency bins
    hidden_dim: int = 512               # Hidden dimension for embeddings and transformers
    n_encoder_layers: int = 6           # Number of transformer encoder layers
    n_decoder_layers: int = 6           # Number of transformer decoder layers
    n_heads: int = 8                    # Number of attention heads
    encoder_ff_dim: int = 2048          # Feed-forward dimension in encoder
    decoder_ff_dim: int = 2048          # Feed-forward dimension in decoder
    encoder_dropout: float = 0.1        # Dropout rate
    max_decoder_seq_len: int = 4000     # Maximum decoder sequence length

    # Loss weights
    duration_loss_weight: float = 0.1   # Weight for duration prediction loss
    stop_token_loss_weight: float = 1.0 # Weight for stop token loss

    # Audio processing parameters (optimized for LJSpeech)
    max_seq_length: int = 2500          # Maximum mel frame sequence length
    sample_rate: int = 22050            # Audio sample rate (LJSpeech is 22050 Hz)
    hop_length: int = 256               # STFT hop length in samples
    win_length: int = 1024              # STFT window length in samples
    n_fft: int = 1024                   # FFT size
    f_min: float = 0.0                  # Minimum frequency
    f_max: float = 8000.0               # Maximum frequency (Nyquist = sr/2 = 11025)

    # Data loading
    num_workers: int = 2                # Number of data loading workers (conservative default)
    # OPTIMIZATION: After first run, monitor GPU utilization. If GPU waits for data:
    # - 4-8 cores: try num_workers=4
    # - 8-16 cores: try num_workers=6
    # - 16+ cores: try num_workers=8
    # Note: Each worker uses ~1-2GB RAM. Monitor with: nvidia-smi dmon -s u
    pin_memory: bool = True             # Pin memory for faster GPU transfer (disable for MPS)
    prefetch_factor: int = 3            # Number of batches to prefetch (only used if num_workers > 0)
    persistent_workers: bool = True     # Keep workers alive between epochs (only used if num_workers > 0)

    # Checkpointing
    save_every: int = 5                 # Save checkpoint every N epochs (increased to save disk space)
    resume_checkpoint: str = 'auto'     # Resume from checkpoint ('auto' for latest, or path to .pth)
    keep_last_n_checkpoints: int = 3    # Only keep the last N checkpoints (auto-delete old ones)

    # Gradient checkpointing (memory optimization)
    gradient_checkpointing: bool = True # Enable gradient checkpointing
    checkpoint_segments: int = 2        # Number of segments for checkpointing
    auto_optimize_checkpointing: bool = True  # Auto-optimize segments based on GPU memory

    # Mixed precision training
    use_mixed_precision: bool = True    # Enable mixed precision (fp16)
    mixed_precision_dtype = torch.float16  # Mixed precision dtype (float16 or bfloat16)
    # OPTIMIZATION: After first successful run, try torch.bfloat16 for more stability
    # mixed_precision_dtype = torch.bfloat16  # Uncomment for bf16 (better stability, requires Ampere+ GPU)
    amp_init_scale: float = 65536.0     # Initial loss scale for AMP
    amp_growth_factor: float = 2.0      # Growth factor for loss scale
    amp_backoff_factor: float = 0.5     # Backoff factor for loss scale
    amp_growth_interval: int = 2000     # Steps between scale increases

    # Gradient clipping
    max_grad_norm: float = 1.0          # Maximum gradient norm for clipping

    # Profiling (debugging)
    enable_profiling: bool = False      # Enable GPU profiling
    profile_epoch_start: int = 1        # Start profiling from this epoch
    profile_wait_steps: int = 1         # Wait steps before profiling
    profile_warmup_steps: int = 1       # Warmup steps for profiling
    profile_steps: int = 5              # Active profiling steps
    run_standalone_profiling: bool = False

    # Interbatch profiling
    enable_interbatch_profiling: bool = False
    interbatch_report_interval: int = 100

    # Adaptive memory management
    enable_adaptive_memory: bool = True
    memory_report_interval: int = 500

    # Logging
    log_dir: str = "runs"               # TensorBoard log directory
    log_interval: int = 50              # Log every N batches

    # Weights & Biases logging
    use_wandb: bool = False             # Enable Weights & Biases logging
    wandb_project: str = "kokoro-english-tts"  # W&B project name
    wandb_entity: Optional[str] = None  # W&B entity (username or team)
    wandb_run_name: Optional[str] = None  # W&B run name (auto-generated if None)
    wandb_tags: list = None             # W&B tags for the run
    wandb_notes: Optional[str] = None   # W&B notes for the run

    # Validation
    validation_split: float = 0.05      # Fraction of data for validation
    validate_every: int = 1             # Validate every N epochs

    def __post_init__(self):
        """Post-initialization validation and adjustments"""
        import os

        # Check if we should suppress output (only during testing)
        quiet = os.environ.get('TESTING')

        # Validate checkpoint segments
        if self.checkpoint_segments < 1:
            self.checkpoint_segments = 1
            if not quiet:
                print("Warning: checkpoint_segments must be >= 1, setting to 1")

        # Disable pin_memory for MPS (not supported)
        if self.device == "mps":
            self.pin_memory = False
            if not quiet:
                print("Note: pin_memory disabled for MPS device")

        # Auto-optimize checkpointing if requested
        if self.auto_optimize_checkpointing and self.gradient_checkpointing:
            self._optimize_checkpointing()

        # Log configuration
        self._log_config()

    def _optimize_checkpointing(self):
        """Optimize checkpoint segments based on available GPU memory"""
        import os

        quiet = os.environ.get('TESTING')

        device = None
        if torch.cuda.is_available():
            device = "cuda"
            if not quiet:
                print("CUDA available, optimizing checkpointing for GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            if not quiet:
                print("MPS available, optimizing checkpointing for Apple Silicon")
        else:
            if not quiet:
                print("No GPU acceleration available, skipping checkpointing optimization")
            return

        try:
            if device == "cuda":
                # Get GPU memory info
                total_memory_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                if not quiet:
                    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                    print(f"Total GPU Memory: {total_memory_mb:.1f} MB")

                # Estimate segments based on available memory
                # More memory = fewer segments needed
                if total_memory_mb > 20000:  # >20GB
                    self.checkpoint_segments = 2
                elif total_memory_mb > 10000:  # >10GB
                    self.checkpoint_segments = 3
                elif total_memory_mb > 6000:   # >6GB
                    self.checkpoint_segments = 4
                else:
                    self.checkpoint_segments = 6  # <6GB - more aggressive

            elif device == "mps":
                # For MPS, use conservative settings
                # MPS unified memory handling is different from CUDA
                if not quiet:
                    print("Using conservative checkpointing settings for MPS")
                self.checkpoint_segments = 4

            if not quiet:
                print(f"Optimized checkpoint_segments: {self.checkpoint_segments}")

        except Exception as e:
            if not quiet:
                print(f"Error optimizing checkpointing: {e}")
                print("Using default checkpoint_segments")

    def _log_config(self):
        """Log important configuration details"""
        import os
        # Skip logging during tests
        if os.environ.get('TESTING'):
            return

        print("\n" + "="*60)
        print("English TTS Training Configuration")
        print("="*60)
        print(f"Dataset: {self.data_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Mixed Precision: {self.use_mixed_precision}")

        if self.gradient_checkpointing:
            print(f"Gradient Checkpointing: Enabled ({self.checkpoint_segments} segments)")
            estimated_savings = (self.checkpoint_segments - 1) / self.checkpoint_segments * 100
            print(f"  Estimated memory savings: ~{estimated_savings:.1f}%")
        else:
            print("Gradient Checkpointing: Disabled")

        print(f"\nAudio Config:")
        print(f"  Sample Rate: {self.sample_rate} Hz")
        print(f"  Mel Channels: {self.n_mels}")
        print(f"  FFT Size: {self.n_fft}")
        print(f"  Hop Length: {self.hop_length}")
        print(f"  Window Length: {self.win_length}")

        print(f"\nModel Config:")
        print(f"  Hidden Dim: {self.hidden_dim}")
        print(f"  Encoder Layers: {self.n_encoder_layers}")
        print(f"  Decoder Layers: {self.n_decoder_layers}")
        print(f"  Attention Heads: {self.n_heads}")
        print(f"  Encoder FF Dim: {self.encoder_ff_dim}")
        print(f"  Decoder FF Dim: {self.decoder_ff_dim}")
        print("="*60 + "\n")

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EnglishTrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


def get_default_config() -> EnglishTrainingConfig:
    """Get default configuration for LJSpeech training"""
    return EnglishTrainingConfig()


def get_small_config() -> EnglishTrainingConfig:
    """Get configuration for testing with smaller model"""
    config = EnglishTrainingConfig(
        batch_size=8,
        n_encoder_layers=4,
        n_decoder_layers=4,
        hidden_dim=256,
        encoder_ff_dim=1024,
        decoder_ff_dim=1024,
        num_epochs=10,
    )
    return config


def get_large_config() -> EnglishTrainingConfig:
    """Get configuration for larger model (requires more GPU memory)"""
    config = EnglishTrainingConfig(
        batch_size=32,
        n_encoder_layers=8,
        n_decoder_layers=8,
        hidden_dim=768,
        encoder_ff_dim=3072,
        decoder_ff_dim=3072,
        n_heads=12,
    )
    return config


if __name__ == "__main__":
    # Test configurations
    print("Default Config:")
    config = get_default_config()

    print("\nSmall Config (for testing):")
    config_small = get_small_config()

    print("\nLarge Config (for high-end GPUs):")
    config_large = get_large_config()
