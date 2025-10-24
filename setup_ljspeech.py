#!/usr/bin/env python3
"""
Setup script for LJSpeech dataset
Downloads dataset and optionally runs Montreal Forced Aligner
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset URLs
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJSPEECH_ZENODO_URL = "https://zenodo.org/records/7499098/files/LJSpeech-1.1.tar.bz2"

LJSPEECH_DIR = "LJSpeech-1.1"
LJSPEECH_ARCHIVE = "LJSpeech-1.1.tar.bz2"


def check_command_exists(command: str) -> bool:
    """Check if a command exists in PATH"""
    try:
        subprocess.run(
            [command, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        return True
    except FileNotFoundError:
        return False


def download_ljspeech(output_dir: str = ".", use_zenodo: bool = False):
    """Download LJSpeech dataset

    Args:
        output_dir: Directory to download to
        use_zenodo: If True, download pre-aligned version from Zenodo (3.8GB with MFA alignments)
                   If False, download original version (2.6GB, no alignments)
    """
    output_path = Path(output_dir)
    archive_path = output_path / LJSPEECH_ARCHIVE
    dataset_path = output_path / LJSPEECH_DIR

    # Check if already downloaded
    if dataset_path.exists():
        logger.info(f"LJSpeech dataset already exists at: {dataset_path}")
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Skipping download")
            return str(dataset_path)

    # Choose URL
    if use_zenodo:
        url = LJSPEECH_ZENODO_URL
        size = "3.8 GB"
        logger.info("Downloading PRE-ALIGNED version from Zenodo")
        logger.info("This includes MFA alignments - no need to run alignment step!")
    else:
        url = LJSPEECH_URL
        size = "2.6 GB"
        logger.info("Downloading original version")
        logger.info("You will need to run MFA alignment afterward for best quality")

    # Download
    logger.info(f"Downloading from: {url}")
    logger.info(f"This is a {size} download - it may take a while...")

    try:
        if check_command_exists("wget"):
            subprocess.run(
                ["wget", "-O", str(archive_path), url],
                check=True
            )
        elif check_command_exists("curl"):
            subprocess.run(
                ["curl", "-L", "-o", str(archive_path), url],
                check=True
            )
        else:
            logger.error("Neither wget nor curl found. Please install one of them.")
            logger.info("Or download manually from: " + url)
            sys.exit(1)

        logger.info("Download complete")

    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)

    # Extract
    logger.info("Extracting archive...")
    try:
        subprocess.run(
            ["tar", "-xjf", str(archive_path), "-C", str(output_path)],
            check=True
        )
        logger.info("Extraction complete")

        # Remove archive to save space
        logger.info("Removing archive to save space...")
        archive_path.unlink()

    except subprocess.CalledProcessError as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)

    logger.info(f"LJSpeech dataset ready at: {dataset_path}")
    return str(dataset_path)


def setup_mfa():
    """Check if MFA is installed and provide setup instructions"""
    logger.info("\nChecking for Montreal Forced Aligner (MFA)...")

    if check_command_exists("mfa"):
        logger.info("MFA is installed!")

        # Check version
        result = subprocess.run(
            ["mfa", "version"],
            capture_output=True,
            text=True
        )
        logger.info(f"MFA version: {result.stdout.strip()}")
        return True

    else:
        logger.warning("Montreal Forced Aligner (MFA) not found")
        logger.info("\nMFA is required for generating phoneme duration alignments.")
        logger.info("Without MFA, the model will use uniform duration fallback (poor quality).")
        logger.info("\nTo install MFA:")
        logger.info("  1. Install conda if not already installed:")
        logger.info("     https://docs.conda.io/en/latest/miniconda.html")
        logger.info("\n  2. Install MFA via conda:")
        logger.info("     conda install -c conda-forge montreal-forced-aligner")
        logger.info("\n  3. Re-run this script with --align flag")
        return False


def run_mfa_alignment(dataset_path: str):
    """Run Montreal Forced Aligner on LJSpeech"""
    logger.info("\nRunning Montreal Forced Aligner...")

    dataset_path = Path(dataset_path)
    output_path = dataset_path / "TextGrid"

    if output_path.exists():
        logger.info(f"Alignments already exist at: {output_path}")
        response = input("Re-run alignment? (y/N): ").strip().lower()
        if response != 'y':
            logger.info("Skipping alignment")
            return str(output_path)

    # Check if MFA is installed
    if not check_command_exists("mfa"):
        logger.error("MFA is not installed. Run setup without --align first.")
        sys.exit(1)

    logger.info("This process will take 1-3 hours depending on your hardware...")

    try:
        # Download acoustic model and dictionary if needed
        logger.info("Downloading English acoustic model and dictionary...")
        subprocess.run(
            ["mfa", "model", "download", "acoustic", "english_us_arpa"],
            check=True
        )
        subprocess.run(
            ["mfa", "model", "download", "dictionary", "english_us_arpa"],
            check=True
        )

        # Run alignment
        logger.info("Running forced alignment...")
        logger.info(f"Input: {dataset_path}")
        logger.info(f"Output: {output_path}")

        # MFA align command
        # mfa align <corpus_dir> <dictionary> <acoustic_model> <output_dir>
        subprocess.run(
            [
                "mfa", "align",
                str(dataset_path),
                "english_us_arpa",  # dictionary
                "english_us_arpa",  # acoustic model
                str(output_path),   # output directory
                "--clean",          # Clean previous runs
                "--verbose"         # Verbose output
            ],
            check=True
        )

        logger.info(f"Alignment complete! TextGrid files saved to: {output_path}")
        return str(output_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"MFA alignment failed: {e}")
        logger.info("\nTroubleshooting:")
        logger.info("  - Make sure conda is activated")
        logger.info("  - Try running MFA commands manually to see detailed errors")
        logger.info("  - Check MFA documentation: https://montreal-forced-aligner.readthedocs.io/")
        sys.exit(1)


def verify_installation(dataset_path: str):
    """Verify the installation"""
    logger.info("\nVerifying installation...")

    dataset_path = Path(dataset_path)

    # Check metadata
    metadata_file = dataset_path / "metadata.csv"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return False

    # Count samples
    with open(metadata_file, 'r', encoding='utf-8') as f:
        num_samples = sum(1 for _ in f)

    logger.info(f"✓ Metadata file: {num_samples} samples")

    # Check wavs
    wavs_dir = dataset_path / "wavs"
    if wavs_dir.exists():
        num_wavs = len(list(wavs_dir.glob("*.wav")))
        logger.info(f"✓ Audio files: {num_wavs} WAV files")
    else:
        logger.error("✗ Audio directory not found")
        return False

    # Check alignments
    textgrid_dir = dataset_path / "TextGrid"
    if textgrid_dir.exists():
        num_textgrids = len(list(textgrid_dir.glob("*.TextGrid")))
        logger.info(f"✓ MFA alignments: {num_textgrids} TextGrid files")
    else:
        logger.warning("✗ No MFA alignments found (will use uniform fallback)")

    logger.info("\nDataset structure:")
    logger.info(f"  {dataset_path}/")
    logger.info(f"    metadata.csv ({num_samples} entries)")
    logger.info(f"    wavs/ ({num_wavs} files)")
    if textgrid_dir.exists():
        logger.info(f"    TextGrid/ ({num_textgrids} files)")

    logger.info("\n✓ Dataset is ready for training!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup LJSpeech dataset for Kokoro English TTS training"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to download dataset to (default: current directory)'
    )

    parser.add_argument(
        '--zenodo',
        action='store_true',
        help='Download pre-aligned version from Zenodo (3.8GB with MFA alignments)'
    )

    parser.add_argument(
        '--align',
        action='store_true',
        help='Run Montreal Forced Aligner after download (not needed if using --zenodo)'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download (use existing dataset)'
    )

    parser.add_argument(
        '--align-only',
        action='store_true',
        help='Only run alignment (assumes dataset already downloaded)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("LJSpeech Dataset Setup")
    print("="*70 + "\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for conflicting flags
    if args.zenodo and args.align:
        logger.warning("--zenodo includes pre-aligned data, --align is not needed")
        logger.info("Proceeding with Zenodo download (alignments already included)")

    dataset_path = None

    # Download dataset
    if not args.skip_download and not args.align_only:
        dataset_path = download_ljspeech(str(output_dir), use_zenodo=args.zenodo)
    else:
        dataset_path = str(output_dir / LJSPEECH_DIR)
        if not Path(dataset_path).exists():
            logger.error(f"Dataset not found at: {dataset_path}")
            logger.info("Run without --skip-download to download it")
            sys.exit(1)

    # Setup/check MFA (skip if using Zenodo with pre-aligned data)
    if args.zenodo and not args.align_only:
        logger.info("\nSkipping MFA setup (using pre-aligned Zenodo data)")
        mfa_installed = False
    else:
        mfa_installed = setup_mfa()

    # Run alignment if requested (but warn if using Zenodo)
    if args.align or args.align_only:
        if args.zenodo and not args.align_only:
            logger.warning("Zenodo dataset already includes alignments - skipping MFA")
        elif not mfa_installed:
            logger.error("Cannot run alignment - MFA is not installed")
            sys.exit(1)
        else:
            run_mfa_alignment(dataset_path)

    # Verify
    verify_installation(dataset_path)

    # Next steps
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)

    if not Path(dataset_path).joinpath("TextGrid").exists():
        print("\n⚠️  No MFA alignments found!")
        print("\nFor better quality, run alignment:")
        print(f"  python setup_ljspeech.py --align-only")
        print("\nOr train without alignments (lower quality):")
        print(f"  python training_english.py --corpus {dataset_path}")

    else:
        print("\n✓ Dataset is ready with MFA alignments!")
        print("\nStart training:")
        print(f"  python training_english.py --corpus {dataset_path}")

    print("\nTest mode (quick test with small subset):")
    print(f"  python training_english.py --corpus {dataset_path} --test-mode")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
