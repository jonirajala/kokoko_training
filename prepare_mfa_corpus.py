#!/usr/bin/env python3
"""
Prepare LJSpeech corpus for MFA alignment by creating .txt files
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_ljspeech_for_mfa(dataset_path: Path):
    """
    Create .txt transcript files for each audio file in LJSpeech
    MFA requires: audio.wav + audio.txt in same directory
    """
    logger.info("Preparing LJSpeech corpus for MFA...")

    metadata_file = dataset_path / "metadata.csv"
    wavs_dir = dataset_path / "wavs"

    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False

    if not wavs_dir.exists():
        logger.error(f"Wavs directory not found: {wavs_dir}")
        return False

    # Read metadata
    transcripts = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 3:
                file_id = parts[0]
                normalized_text = parts[2]  # Use normalized text
                transcripts[file_id] = normalized_text

    logger.info(f"Loaded {len(transcripts)} transcripts from metadata")

    # Create .txt files
    created = 0
    skipped = 0

    for file_id, text in transcripts.items():
        txt_path = wavs_dir / f"{file_id}.txt"

        # Skip if already exists
        if txt_path.exists():
            skipped += 1
            continue

        # Write transcript
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        created += 1

        if created % 1000 == 0:
            logger.info(f"Created {created} transcript files...")

    logger.info(f"\nDone!")
    logger.info(f"  Created: {created} new transcript files")
    logger.info(f"  Skipped: {skipped} existing files")
    logger.info(f"  Total: {len(transcripts)} transcripts")

    return True


if __name__ == "__main__":
    dataset_path = Path("LJSpeech-1.1")

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        exit(1)

    success = prepare_ljspeech_for_mfa(dataset_path)

    if success:
        logger.info("\n✓ Corpus is ready for MFA alignment!")
        logger.info("Now run: python setup_ljspeech.py --align-only")
    else:
        logger.error("\n✗ Failed to prepare corpus")
        exit(1)
