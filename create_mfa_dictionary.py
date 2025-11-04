#!/usr/bin/env python3
"""
Create a custom MFA dictionary that matches the Misaki G2P phoneme output
This allows MFA to align using the same phoneme set as training
"""

import sys
from pathlib import Path
import logging
from typing import Dict, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from data.english_phoneme_processor import EnglishPhonemeProcessor


def create_mfa_dictionary_from_dataset(dataset_path: Path, output_dict_path: Path):
    """
    Create an MFA-compatible dictionary from the actual G2P output of the dataset

    This extracts all unique word->phoneme mappings from running Misaki G2P
    on the LJSpeech text, then formats them for MFA.
    """
    logger.info("="*80)
    logger.info("Creating MFA Dictionary from Misaki G2P Output")
    logger.info("="*80)

    # Load phoneme processor
    processor = EnglishPhonemeProcessor(variant='en-us')

    if not processor.use_misaki:
        logger.error("Misaki G2P is not available!")
        logger.error("Install with: pip install 'misaki[en]'")
        sys.exit(1)

    logger.info(f"\nProcessing LJSpeech dataset at: {dataset_path}")

    # Load metadata
    metadata_file = dataset_path / "metadata.csv"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        sys.exit(1)

    # Collect unique word->phoneme mappings
    word_to_phonemes: Dict[str, Set[str]] = {}

    logger.info("\nProcessing texts to extract word->phoneme mappings...")

    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 1000 == 0:
                logger.info(f"  Processed {line_num} lines...")

            parts = line.strip().split('|')
            if len(parts) >= 3:
                text = parts[2]  # normalized text

                try:
                    # Get phonemes from Misaki
                    ipa_text, tokens = processor.g2p(text)
                    phonemes = processor._parse_ipa(ipa_text)

                    # Extract words from tokens
                    # Misaki returns tokens with phoneme boundaries
                    # We need to align words to phoneme sequences
                    words = text.lower().split()

                    # Simple heuristic: assign phonemes to words based on text position
                    # This is approximate but works for dictionary building
                    for word in words:
                        # Clean word
                        word_clean = ''.join(c for c in word if c.isalnum() or c in ["'", "-"])
                        if not word_clean:
                            continue

                        # Store word (we'll get phonemes from full G2P later)
                        if word_clean not in word_to_phonemes:
                            # Process word in isolation to get its phonemes
                            try:
                                word_ipa, _ = processor.g2p(word_clean)
                                word_phonemes = processor._parse_ipa(word_ipa)
                                # Clean stress markers and punctuation for MFA
                                word_phonemes_clean = [p for p in word_phonemes
                                                      if p not in ['ˈ', 'ˌ', '.', ',', '!', '?', '-', '—', ':', ';', '"', "'", '(', ')']]
                                if word_phonemes_clean:
                                    phoneme_str = ' '.join(word_phonemes_clean)
                                    if word_clean not in word_to_phonemes:
                                        word_to_phonemes[word_clean] = set()
                                    word_to_phonemes[word_clean].add(phoneme_str)
                            except:
                                pass

                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

    logger.info(f"\nCollected {len(word_to_phonemes)} unique words")

    # Write MFA dictionary format
    logger.info(f"\nWriting MFA dictionary to: {output_dict_path}")

    with open(output_dict_path, 'w', encoding='utf-8') as f:
        for word in sorted(word_to_phonemes.keys()):
            for phoneme_seq in sorted(word_to_phonemes[word]):
                f.write(f"{word}\t{phoneme_seq}\n")

    logger.info(f"✓ Dictionary created with {sum(len(v) for v in word_to_phonemes.values())} entries")

    return output_dict_path


def main():
    logger.info("="*80)
    logger.info("MFA Dictionary Creator for Misaki G2P")
    logger.info("="*80)

    dataset_path = Path("LJSpeech-1.1")
    output_dict_path = Path("misaki_mfa_dictionary.dict")

    if not dataset_path.exists():
        logger.error(f"Dataset not found at: {dataset_path}")
        logger.error("Download first with: python setup_ljspeech.py")
        sys.exit(1)

    create_mfa_dictionary_from_dataset(dataset_path, output_dict_path)

    logger.info("\n" + "="*80)
    logger.info("Next Steps")
    logger.info("="*80)
    logger.info("\n1. Run MFA alignment with the custom dictionary:")
    logger.info(f"   mfa align LJSpeech-1.1 {output_dict_path} \\")
    logger.info("     english_us_arpa LJSpeech-1.1/TextGrid --clean")
    logger.info("\n2. Or update setup_ljspeech.py to use this dictionary automatically")
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
