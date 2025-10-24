# Training Infrastructure

This folder contains all training-related components including trainers, configuration, checkpointing, and optimization utilities.

## Files

### `config_english.py` (Configuration)
**Purpose**: Training configuration dataclass
**Key Functions**:
- `EnglishTrainingConfig` - Main config dataclass
- `get_default_config()` - Standard configuration
- `get_small_config()` - Smaller model for testing

**Configuration Categories**:
1. **Data**: Paths, batch size, workers
2. **Model**: Architecture dimensions
3. **Training**: Learning rate, epochs, optimizer
4. **Audio**: Sample rate, mel parameters
5. **Hardware**: Device, mixed precision, memory
6. **Logging**: W&B, checkpoints, profiling

### `trainer.py` (Base Trainer)
**Purpose**: Core training loop with profiling and memory management
**Key Components**:
- `KokoroTrainer` - Base trainer class
- Training loop with progress bars
- Mixed precision support (CUDA/MPS)
- Adaptive memory management
- Profiling and benchmarking

**Features**:
- Automatic checkpoint resumption
- Learning rate scheduling (CosineAnnealingWarmRestarts)
- Gradient clipping (norm=1.0)
- OOM recovery
- Periodic memory cleanup

### `english_trainer.py` (English Trainer)
**Purpose**: Extends base trainer for English TTS with W&B logging
**Additions**:
- Per-batch W&B logging (every 10 batches)
- Epoch summary metrics
- Memory and mixed precision tracking
- Smooth loss curves in W&B dashboard

**W&B Metrics**:
- Batch losses (total, mel, duration, stop)
- Learning rate schedule
- Gradient scale (mixed precision)
- Memory pressure
- Throughput (samples/sec)

### `checkpoint_manager.py` (Checkpointing)
**Purpose**: Save and load model checkpoints
**Key Functions**:
- `save_checkpoint()` - Save training state
- `load_checkpoint()` - Resume from checkpoint
- `save_phoneme_processor()` - Save vocab
- `find_latest_checkpoint()` - Auto-resume

**Checkpoint Contents**:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'loss': float,
    'config': TrainingConfig,
    'scaler': dict  # Mixed precision state
}
```

### `adaptive_memory_manager.py` (Memory Optimization)
**Purpose**: Intelligent memory cleanup based on pressure
**Key Components**:
- `MemoryPressureLevel` - Low/Moderate/High/Critical
- `AdaptiveMemoryManager` - Cleanup coordinator
- Device-specific thresholds (CUDA vs MPS)

**Features**:
- Monitors memory usage every batch
- Triggers cleanup when pressure high
- Tracks cleanup overhead
- Emergency cleanup on OOM
- Device-aware (CUDA/MPS different strategies)

**Thresholds (CUDA)**:
- Low: < 60% used
- Moderate: 60-75% used
- High: 75-85% used
- Critical: > 85% used

### `interbatch_profiler.py` (Performance Profiling)
**Purpose**: Measure time spent between batches
**Metrics**:
- Data loading time
- Forward pass time
- Backward pass time
- Interbatch gap (waiting time)
- Throughput (samples/sec)

**Use Case**: Identify bottlenecks in training pipeline

### `mps_grad_scaler.py` (MPS Mixed Precision)
**Purpose**: Custom gradient scaler for Apple Silicon (MPS)
**Why Needed**: PyTorch's built-in scaler is CUDA-only
**Features**:
- Loss scaling for FP16 training
- Overflow detection
- Dynamic scale adjustment
- Compatible with MPS backend

### `device_type.py` (Device Enumeration)
**Purpose**: Simple enum for device types
**Values**: `CUDA`, `MPS`, `CPU`

## Training Flow

### Initialization:
```python
1. Load config
2. Create dataset & dataloader
3. Initialize model
4. Setup optimizer & scheduler
5. Load checkpoint (if resuming)
6. Initialize W&B (if enabled)
```

### Training Loop:
```python
for epoch in range(start_epoch, num_epochs):
    for batch in dataloader:
        # 1. Adaptive memory check
        cleanup_result = memory_manager.adaptive_cleanup(batch_idx)

        # 2. Load data to device
        mel_specs, phonemes, durations = batch

        # 3. Forward pass (with mixed precision)
        with autocast():
            mel_pred, dur_pred, stop_pred = model(...)
            loss = criterion(mel_pred, mel_specs, ...)

        # 4. Backward pass
        scaler.scale(loss).backward()

        # 5. Optimizer step
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # 6. Log to W&B (every 10 batches)
        if batch_idx % 10 == 0:
            wandb.log({
                'train/batch_total_loss': loss.item(),
                ...
            })

    # 7. Epoch end
    scheduler.step()
    save_checkpoint(epoch, loss)
    wandb.log({'train/epoch_total_loss': avg_loss})
```

## Configuration Examples

### Standard Training:
```python
config = get_default_config()
# - Batch size: 16
# - Hidden dim: 512
# - Encoder/Decoder: 6 layers each
# - Learning rate: 1e-4
# - Mixed precision: Enabled
```

### Small Model (Testing):
```python
config = get_small_config()
# - Batch size: 8
# - Hidden dim: 256
# - Encoder/Decoder: 4 layers each
# - Faster training for debugging
```

### Custom Configuration:
```python
config = EnglishTrainingConfig(
    data_dir="LJSpeech-1.1",
    batch_size=32,
    learning_rate=5e-5,
    hidden_dim=768,
    n_encoder_layers=8,
    use_wandb=True,
    enable_profiling=False
)
```

## Key Features

### Mixed Precision Training:
- **CUDA**: Native `torch.cuda.amp.GradScaler`
- **MPS**: Custom `MPSGradScaler` (Apple Silicon)
- **Benefits**: 30-50% faster, 40-60% less memory
- **Fallback**: Disabled on CPU or if errors occur

### Adaptive Memory Management:
- Monitors GPU/MPS memory every batch
- Cleans up cache when pressure detected
- Prevents OOM errors proactively
- Tracks overhead (<1% typical)

### Weights & Biases Integration:
- Automatic experiment tracking
- Real-time loss curves
- System metrics (GPU, memory)
- Hyperparameter logging
- Model checkpointing to cloud

### Checkpoint Management:
- Auto-resume with `--resume auto`
- Saves every N epochs (configurable)
- Includes full training state
- Model selection by lowest loss

## Performance Tips

### Memory Optimization:
1. **Gradient Checkpointing**: Enabled by default (75% memory reduction)
2. **Adaptive Cleanup**: Prevents OOM without sacrificing speed
3. **Batch Size**: Start large, reduce if OOM
4. **Mixed Precision**: Use `--no-mixed-precision` if unstable

### Speed Optimization:
1. **Data Loading**: Use 2-4 workers (not more)
2. **Pin Memory**: Enabled for CUDA automatically
3. **Length-Based Batching**: Reduces padding waste
4. **Profiling**: Disable after debugging (`enable_profiling=False`)

### Quality Optimization:
1. **Learning Rate**: 1e-4 is good default
2. **Scheduler**: Cosine annealing with warm restarts
3. **Gradient Clipping**: Prevents exploding gradients
4. **MFA Alignments**: Essential for duration accuracy

## Dependencies

- `torch` - PyTorch training
- `wandb` - Experiment tracking (optional)
- `tqdm` - Progress bars
- `data.ljspeech_dataset` - Dataset
- `kokoro.model` - Model architecture

## Design Principles

1. **Modularity**: Separate concerns (training, logging, memory)
2. **Robustness**: Graceful degradation, error recovery
3. **Observability**: Comprehensive logging and metrics
4. **Efficiency**: Mixed precision, adaptive memory
5. **Simplicity**: Clean APIs, minimal configuration

## Common Issues

### OOM Errors:
- Reduce batch size
- Enable gradient checkpointing
- Disable mixed precision
- Use adaptive memory management

### Slow Training:
- Check data loading bottleneck (profiler)
- Increase batch size if memory available
- Use mixed precision
- Reduce number of workers if CPU-bound

### NaN Losses:
- Check learning rate (too high?)
- Disable mixed precision temporarily
- Check input data (NaN values?)
- Review gradient clipping

## Notes

- Checkpoints saved every 5 epochs by default
- W&B logging optional but recommended
- Mixed precision tested on CUDA and MPS
- Adaptive memory manager works on all devices
- Profiling adds ~5% overhead when enabled
