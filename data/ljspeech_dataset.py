#!/usr/bin/env python3
"""
LJSpeech Dataset implementation with Montreal Forced Aligner support
"""

import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
import logging
import random
import numpy as np
from tqdm import tqdm

# Set torchaudio backend to avoid torchcodec dependency issues
# Use soundfile backend which is more stable
torchaudio.set_audio_backend("soundfile")

logger = logging.getLogger(__name__)


class LJSpeechDataset(Dataset):
    """
    Dataset class for LJSpeech corpus with MFA alignments.

    This dataset loads audio, text, and phoneme duration alignments from
    the LJSpeech dataset processed with Montreal Forced Aligner.
    """

    def __init__(self, data_dir: str, config):
        from .english_phoneme_processor import EnglishPhonemeProcessor

        self.data_dir = Path(data_dir)
        self.config = config
        self.phoneme_processor = EnglishPhonemeProcessor(variant='en-us')

        # TextGrid files from MFA (optional)
        # Note: MFA outputs to TextGrid/wavs/ subdirectory
        self.alignment_dir = self.data_dir / "TextGrid" / "wavs"
        self.has_alignments = self.alignment_dir.exists()

        if self.has_alignments:
            logger.info(f"Found MFA alignments at: {self.alignment_dir}")
        else:
            raise ValueError(f"No MFA alignments found at {self.alignment_dir}")
        # Validate MelSpectrogram parameters
        if self.config.win_length > self.config.n_fft:
            raise ValueError(
                f"win_length ({self.config.win_length}) cannot be greater than n_fft ({self.config.n_fft})"
            )
        if self.config.hop_length <= 0:
            raise ValueError("hop_length must be a positive integer")

        # Pre-create mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
            power=2.0,
            normalized=False,
            window_fn=torch.hann_window,
        )

        # Load metadata
        self.samples = self._load_samples()
        self.skipped_samples = 0  # Track skipped samples due to errors
        logger.info(f"Loaded {len(self.samples)} samples from LJSpeech at {data_dir}")

    def _load_samples(self) -> List[Dict]:
        """
        Load samples from LJSpeech metadata.

        LJSpeech format: filename|original_text|normalized_text
        """
        samples = []

        metadata_file = self.data_dir / "metadata.csv"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_file}\n"
                f"Expected LJSpeech structure:\n"
                f"  {self.data_dir}/\n"
                f"    metadata.csv\n"
                f"    wavs/\n"
                f"    TextGrid/ (optional, from MFA)"
            )

        logger.info(f"Loading metadata from {metadata_file}")

        # Count lines for progress bar
        with open(metadata_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading metadata"):
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    audio_file_stem = parts[0]

                    # Read normalized text from .txt file (created during MFA setup with number normalization)
                    txt_path = self.data_dir / "wavs" / f"{audio_file_stem}.txt"
                    if txt_path.exists():
                        with open(txt_path, 'r', encoding='utf-8') as txt_f:
                            normalized_text = txt_f.read().strip()
                    else:
                        # Fallback to metadata.csv if .txt doesn't exist
                        normalized_text = parts[2] if len(parts) >= 3 else parts[1]

                    audio_path = self.data_dir / "wavs" / f"{audio_file_stem}.wav"

                    if audio_path.exists():
                        sample_dict = {
                            'audio_path': str(audio_path),
                            'text': normalized_text,
                            'audio_file': audio_file_stem,
                        }

                        # Add alignment path if available
                        if self.has_alignments:
                            alignment_path = self.alignment_dir / f"{audio_file_stem}.TextGrid"
                            if alignment_path.exists():
                                sample_dict['alignment_path'] = str(alignment_path)
                            else:
                                sample_dict['alignment_path'] = None
                        else:
                            sample_dict['alignment_path'] = None

                        samples.append(sample_dict)

        return samples

    def _load_mfa_durations(
        self,
        alignment_path: Optional[str],
        phoneme_count: int,
        mel_frame_count: int
    ) -> torch.Tensor:
        """
        Load real phoneme durations from MFA TextGrid file.

        NO SILENT FALLBACKS - Will raise errors if MFA alignments are missing or invalid.
        This ensures you know immediately if there's a problem with alignments.

        Args:
            alignment_path: Path to TextGrid alignment file
            phoneme_count: Number of phonemes (for validation)
            mel_frame_count: Total mel frames (for validation)

        Returns:
            Tensor of phoneme durations (in mel frames)

        Raises:
            ValueError: If alignment file is missing or invalid
            ImportError: If textgrid library is not installed
        """
        # NO FALLBACK: Raise error if alignment path is missing
        if alignment_path is None:
            raise ValueError(
                "MFA alignment path is None! This should never happen.\n"
                "Check that alignment_path is set correctly in dataset samples."
            )

        if not Path(alignment_path).exists():
            raise FileNotFoundError(
                f"MFA alignment file not found: {alignment_path}\n"
                f"Run: python setup_ljspeech.py --align-only\n"
                f"to generate proper IPA alignments matching Misaki G2P."
            )

        # NO FALLBACK: Raise error if textgrid library missing
        try:
            import textgrid
        except ImportError as e:
            raise ImportError(
                "textgrid library not installed!\n"
                "Install with: pip install textgrid\n"
                f"Original error: {e}"
            )

        # Load TextGrid file
        tg = textgrid.TextGrid.fromFile(alignment_path)

        # Find the phones tier (MFA usually names it 'phones' or 'phonemes')
        phones_tier = None
        for tier in tg.tiers:
            if tier.name.lower() in ['phones', 'phone', 'phonemes', 'phoneme']:
                phones_tier = tier
                break

        # NO FALLBACK: Raise error if no phones tier
        if phones_tier is None:
            available_tiers = [t.name for t in tg.tiers]
            raise ValueError(
                f"No phones tier found in {alignment_path}!\n"
                f"Available tiers: {available_tiers}\n"
                f"MFA should create a 'phones' tier. Re-run alignment."
            )

        # Extract durations in seconds
        durations_sec = []
        for interval in phones_tier:
            # Skip empty intervals and silence markers
            if interval.mark and interval.mark.strip() and interval.mark.lower() not in ['sil', 'sp', '']:
                duration = interval.maxTime - interval.minTime
                durations_sec.append(duration)

        # Convert to mel frames
        # mel_frames = duration_sec * sample_rate / hop_length
        durations_frames = [
            int(d * self.config.sample_rate / self.config.hop_length)
            for d in durations_sec
        ]

        # Ensure minimum duration of 1 frame
        durations_frames = [max(1, d) for d in durations_frames]

        # Validate length matches phonemes (approximately)
        # Now using ARPA (g2p_en) which matches MFA's ARPA perfectly!
        # Tolerance set to 30% to handle:
        # 1. OOV words not in MFA dictionary (rare/technical terms)
        # 2. Parenthetical text marked as [bracketed] by MFA
        # 3. Different pronunciations between g2p_en and MFA dictionary
        # Text normalization (numbers → words) handled most issues
        mismatch_pct = abs(len(durations_frames) - phoneme_count) / phoneme_count if phoneme_count > 0 else 1.0

        if mismatch_pct > 0.30:
            raise ValueError(
                f"Phoneme count mismatch in {alignment_path}!\n"
                f"MFA phonemes: {len(durations_frames)}, G2P phonemes: {phoneme_count}\n"
                f"Mismatch: {mismatch_pct*100:.1f}% (threshold: 30%)\n"
                f"\n"
                f"This means your MFA alignments (ARPA) don't match g2p_en (ARPA) phoneme set.\n"
                f"This is unusual since both use ARPA phonemes.\n"
                f"\n"
                f"Possible solutions:\n"
                f"1. Increase tolerance to 20-25% in data/ljspeech_dataset.py line 219\n"
                f"2. Re-generate alignments: python setup_ljspeech.py --align-only\n"
                f"3. Check if alignments are from correct source (should be ARPA-based)"
            )

        # Adjust durations to match phoneme count if needed (minor differences)
        if len(durations_frames) != phoneme_count:
            logger.debug(
                f"Minor phoneme count difference ({len(durations_frames)} vs {phoneme_count}), "
                f"adjusting durations..."
            )
            durations_frames = self._adjust_duration_count(
                durations_frames,
                phoneme_count,
                mel_frame_count
            )

        return torch.tensor(durations_frames, dtype=torch.long)

    def _adjust_duration_count(
        self,
        durations: List[int],
        target_count: int,
        total_frames: int
    ) -> List[int]:
        """
        Adjust duration list to match target phoneme count.

        Args:
            durations: Original duration list
            target_count: Target number of durations
            total_frames: Total mel frames available

        Returns:
            Adjusted duration list
        """
        current_count = len(durations)

        if current_count == target_count:
            return durations

        if current_count < target_count:
            # Need to add more durations - split longest durations
            while len(durations) < target_count:
                max_idx = durations.index(max(durations))
                max_val = durations[max_idx]
                # Split into two roughly equal parts
                durations[max_idx] = max_val // 2
                durations.insert(max_idx + 1, max_val - max_val // 2)

        else:
            # Need to reduce durations - merge shortest adjacent pairs
            while len(durations) > target_count:
                # Find smallest adjacent sum
                min_sum = float('inf')
                min_idx = 0
                for i in range(len(durations) - 1):
                    if durations[i] + durations[i+1] < min_sum:
                        min_sum = durations[i] + durations[i+1]
                        min_idx = i
                # Merge
                durations[min_idx] = durations[min_idx] + durations[min_idx + 1]
                durations.pop(min_idx + 1)

        return durations

    def _generate_uniform_durations(
        self,
        num_phonemes: int,
        num_mel_frames: int
    ) -> torch.Tensor:
        """
        Generate uniform durations as fallback when MFA alignments not available.

        Args:
            num_phonemes: Number of phonemes
            num_mel_frames: Total mel frames

        Returns:
            Tensor of durations
        """
        if num_phonemes == 0:
            return torch.zeros(0, dtype=torch.long)

        # Distribute frames uniformly across phonemes
        avg_duration = num_mel_frames / num_phonemes
        durations = torch.full((num_phonemes,), int(avg_duration), dtype=torch.long)

        # Distribute remainder frames to early phonemes
        remainder = num_mel_frames - torch.sum(durations).item()
        for i in range(min(remainder, num_phonemes)):
            durations[i] += 1

        # Ensure minimum duration of 1
        durations = torch.clamp(durations, min=1)

        return durations

    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                - phoneme_indices: Tensor of phoneme IDs
                - mel_spec: Mel spectrogram [time, n_mels]
                - phoneme_durations: Duration for each phoneme in mel frames
                - stop_token_targets: Binary stop token targets
                - audio_file: Audio filename
                - text: Original text
        """
        sample = self.samples[idx]

        try:
            # Load audio - TorchCodec handles this automatically
            waveform, sr = torchaudio.load(sample['audio_path'])
            if sr != self.config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.sample_rate)
                waveform = resampler(waveform)

            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Ensure minimum length for STFT
            if waveform.shape[1] < self.config.win_length:
                padding_needed = self.config.win_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding_needed))

            # Convert to mel spectrogram
            mel_spec = self.mel_transform(waveform).squeeze(0).T  # [time, n_mels]

            # Apply log scaling with proper clamping
            # CRITICAL FIX: Clamp to valid log-mel range [-11.5, 0.0]
            mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
            mel_spec = torch.clamp(mel_spec, min=-11.5, max=0.0)

            # Clip to max sequence length
            max_frames = self.config.max_seq_length
            if mel_spec.shape[0] > max_frames:
                mel_spec = mel_spec[:max_frames, :]

            # Process text to phonemes
            phoneme_indices = self.phoneme_processor.text_to_indices(sample['text'])
            phoneme_indices_tensor = torch.tensor(phoneme_indices, dtype=torch.long)

            # Load durations (real from MFA or uniform fallback)
            phoneme_durations = self._load_mfa_durations(
                sample.get('alignment_path'),
                len(phoneme_indices),
                mel_spec.shape[0]
            )

            # Generate stop token targets
            stop_token_targets = torch.zeros(mel_spec.shape[0], dtype=torch.float32)
            if mel_spec.shape[0] > 0:
                stop_token_targets[-1] = 1.0

            return {
                'phoneme_indices': phoneme_indices_tensor,
                'mel_spec': mel_spec,
                'phoneme_durations': phoneme_durations,
                'stop_token_targets': stop_token_targets,
                'audio_file': sample['audio_file'],
                'text': sample['text']
            }

        except Exception as e:
            # Changed to DEBUG to avoid spamming logs during training
            # Most errors are phoneme mismatches due to OOV words (expected)
            self.skipped_samples += 1
            logger.error(f"ERROR loading sample {sample['audio_file']}: {e}")
            import traceback
            traceback.print_exc()
            # Return a dummy sample to avoid breaking the batch
            return {
                'phoneme_indices': torch.tensor([0], dtype=torch.long),
                'mel_spec': torch.zeros((1, self.config.n_mels), dtype=torch.float32),
                'phoneme_durations': torch.tensor([1], dtype=torch.long),
                'stop_token_targets': torch.tensor([1.0], dtype=torch.float32),
                'audio_file': sample['audio_file'],
                'text': ''
            }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Pads sequences to same length within batch.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with padded tensors
    """
    # Extract sequences
    phoneme_indices_list = [item['phoneme_indices'] for item in batch]
    mel_specs_list = [item['mel_spec'] for item in batch]
    phoneme_durations_list = [item['phoneme_durations'] for item in batch]
    stop_token_targets_list = [item['stop_token_targets'] for item in batch]

    # Get lengths
    phoneme_lengths = torch.tensor([len(p) for p in phoneme_indices_list], dtype=torch.long)
    mel_lengths = torch.tensor([m.shape[0] for m in mel_specs_list], dtype=torch.long)

    # Pad phoneme sequences
    phoneme_indices_padded = pad_sequence(phoneme_indices_list, batch_first=True, padding_value=0)
    phoneme_durations_padded = pad_sequence(phoneme_durations_list, batch_first=True, padding_value=0)

    # Pad mel spectrograms and stop tokens
    max_mel_len = max(m.shape[0] for m in mel_specs_list)
    mel_dim = mel_specs_list[0].shape[1]

    mel_specs_padded = torch.zeros(len(batch), max_mel_len, mel_dim)
    stop_token_targets_padded = torch.zeros(len(batch), max_mel_len)

    for i, (mel, stop) in enumerate(zip(mel_specs_list, stop_token_targets_list)):
        mel_len = mel.shape[0]
        mel_specs_padded[i, :mel_len, :] = mel
        stop_token_targets_padded[i, :mel_len] = stop

    return {
        'phoneme_indices': phoneme_indices_padded,
        'mel_specs': mel_specs_padded,
        'phoneme_durations': phoneme_durations_padded,
        'stop_token_targets': stop_token_targets_padded,
        'phoneme_lengths': phoneme_lengths,
        'mel_lengths': mel_lengths,
        'audio_files': [item['audio_file'] for item in batch],
        'texts': [item['text'] for item in batch]
    }


class LengthBasedBatchSampler(Sampler):
    """
    Batch sampler that groups samples by similar length for efficient GPU utilization.
    """

    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = True, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Pre-calculate lengths
        logger.info("Calculating sample lengths for batch sampler...")
        self.lengths = self._get_lengths()

    def _get_lengths(self) -> List[int]:
        """Get approximate length for each sample"""
        lengths = []
        for sample in tqdm(self.dataset.samples, desc="Calculating lengths"):
            # Use text length as proxy (faster than loading audio)
            text_len = len(sample['text'])
            lengths.append(text_len)
        return lengths

    def __iter__(self):
        # Create indices sorted by length
        indices = list(range(len(self.dataset)))
        lengths_with_idx = list(zip(self.lengths, indices))
        lengths_with_idx.sort()

        # Group into batches
        batches = []
        for i in range(0, len(lengths_with_idx), self.batch_size):
            batch = [idx for _, idx in lengths_with_idx[i:i + self.batch_size]]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # Shuffle batches
        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
