#!/usr/bin/env python3
"""
Analyze MFA TextGrid files to check if alignments are realistic or uniform
"""

import sys
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import textgrid
except ImportError:
    logger.error("textgrid library not installed!")
    logger.error("Install with: pip install textgrid")
    sys.exit(1)


def analyze_textgrid_file(textgrid_path: Path, sample_rate: int = 22050, hop_length: int = 256):
    """
    Analyze a single TextGrid file to extract phoneme duration statistics

    Returns:
        dict with duration statistics or None if error
    """
    try:
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))

        # Find phones tier
        phones_tier = None
        for tier in tg.tiers:
            if tier.name.lower() in ['phones', 'phone', 'phonemes', 'phoneme']:
                phones_tier = tier
                break

        if phones_tier is None:
            logger.warning(f"No phones tier in {textgrid_path.name}")
            return None

        # Extract durations
        durations_sec = []
        phonemes = []

        for interval in phones_tier:
            # Skip silence markers
            if interval.mark and interval.mark.strip() and interval.mark.lower() not in ['sil', 'sp', '', 'spn']:
                duration_sec = interval.maxTime - interval.minTime
                durations_sec.append(duration_sec)
                phonemes.append(interval.mark)

        if not durations_sec:
            return None

        # Convert to frames
        durations_frames = [d * sample_rate / hop_length for d in durations_sec]

        return {
            'filename': textgrid_path.name,
            'num_phonemes': len(durations_sec),
            'durations_sec': durations_sec,
            'durations_frames': durations_frames,
            'phonemes': phonemes,
            'mean_dur_sec': np.mean(durations_sec),
            'std_dur_sec': np.std(durations_sec),
            'min_dur_sec': np.min(durations_sec),
            'max_dur_sec': np.max(durations_sec),
            'mean_dur_frames': np.mean(durations_frames),
            'std_dur_frames': np.std(durations_frames),
            'min_dur_frames': np.min(durations_frames),
            'max_dur_frames': np.max(durations_frames)
        }

    except Exception as e:
        logger.warning(f"Error analyzing {textgrid_path.name}: {e}")
        return None


def analyze_phoneme_specific_durations(results):
    """
    Analyze if different phoneme types have different durations
    (vowels should be longer than plosives, etc.)
    """
    # Common phoneme categories
    vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
    plosives = ['B', 'D', 'G', 'K', 'P', 'T']
    fricatives = ['DH', 'F', 'S', 'SH', 'TH', 'V', 'Z', 'ZH', 'HH']
    nasals = ['M', 'N', 'NG']

    vowel_durs = []
    plosive_durs = []
    fricative_durs = []
    nasal_durs = []

    for result in results:
        if result is None:
            continue

        for phoneme, dur_sec in zip(result['phonemes'], result['durations_sec']):
            # Strip stress markers (0, 1, 2)
            phoneme_base = phoneme.rstrip('012')

            if phoneme_base in vowels:
                vowel_durs.append(dur_sec)
            elif phoneme_base in plosives:
                plosive_durs.append(dur_sec)
            elif phoneme_base in fricatives:
                fricative_durs.append(dur_sec)
            elif phoneme_base in nasals:
                nasal_durs.append(dur_sec)

    logger.info("\n" + "="*80)
    logger.info("PHONEME-SPECIFIC DURATION ANALYSIS")
    logger.info("="*80)

    if vowel_durs:
        logger.info(f"\nVowels (n={len(vowel_durs)}):")
        logger.info(f"  Mean: {np.mean(vowel_durs):.4f}s ({np.mean(vowel_durs)*22050/256:.2f} frames)")
        logger.info(f"  Std:  {np.std(vowel_durs):.4f}s")
        logger.info(f"  Range: [{np.min(vowel_durs):.4f}, {np.max(vowel_durs):.4f}]s")

    if plosive_durs:
        logger.info(f"\nPlosives (n={len(plosive_durs)}):")
        logger.info(f"  Mean: {np.mean(plosive_durs):.4f}s ({np.mean(plosive_durs)*22050/256:.2f} frames)")
        logger.info(f"  Std:  {np.std(plosive_durs):.4f}s")
        logger.info(f"  Range: [{np.min(plosive_durs):.4f}, {np.max(plosive_durs):.4f}]s")

    if fricative_durs:
        logger.info(f"\nFricatives (n={len(fricative_durs)}):")
        logger.info(f"  Mean: {np.mean(fricative_durs):.4f}s ({np.mean(fricative_durs)*22050/256:.2f} frames)")
        logger.info(f"  Std:  {np.std(fricative_durs):.4f}s")
        logger.info(f"  Range: [{np.min(fricative_durs):.4f}, {np.max(fricative_durs):.4f}]s")

    if nasal_durs:
        logger.info(f"\nNasals (n={len(nasal_durs)}):")
        logger.info(f"  Mean: {np.mean(nasal_durs):.4f}s ({np.mean(nasal_durs)*22050/256:.2f} frames)")
        logger.info(f"  Std:  {np.std(nasal_durs):.4f}s")
        logger.info(f"  Range: [{np.min(nasal_durs):.4f}, {np.max(nasal_durs):.4f}]s")

    # Check if there's meaningful variance
    if vowel_durs and plosive_durs:
        vowel_mean = np.mean(vowel_durs)
        plosive_mean = np.mean(plosive_durs)
        ratio = vowel_mean / plosive_mean if plosive_mean > 0 else 0

        logger.info(f"\nVowel-to-Plosive Duration Ratio: {ratio:.2f}")

        if ratio < 1.5:
            logger.warning("⚠️  Vowels are not significantly longer than plosives!")
            logger.warning("   Expected ratio: 2.0-3.0x for realistic speech")
            logger.warning("   This suggests MFA alignments may be too uniform")
        else:
            logger.info(f"✓ Good variance: vowels are {ratio:.2f}x longer than plosives")


def main():
    logger.info("="*80)
    logger.info("MFA TEXTGRID ALIGNMENT ANALYSIS")
    logger.info("="*80)

    # Find TextGrid directory
    textgrid_dir = Path("LJSpeech-1.1/TextGrid/wavs")

    if not textgrid_dir.exists():
        logger.error(f"TextGrid directory not found: {textgrid_dir}")
        logger.error("Make sure MFA alignments are in the correct location")
        sys.exit(1)

    # Get all TextGrid files
    textgrid_files = list(textgrid_dir.glob("*.TextGrid"))

    if not textgrid_files:
        logger.error(f"No TextGrid files found in {textgrid_dir}")
        sys.exit(1)

    logger.info(f"Found {len(textgrid_files)} TextGrid files")

    # Analyze first 100 files
    num_to_analyze = min(100, len(textgrid_files))
    logger.info(f"Analyzing first {num_to_analyze} files...\n")

    results = []
    for i, tg_file in enumerate(textgrid_files[:num_to_analyze]):
        if (i + 1) % 25 == 0:
            logger.info(f"Processed {i+1}/{num_to_analyze} files...")

        result = analyze_textgrid_file(tg_file)
        if result:
            results.append(result)

    logger.info(f"\nSuccessfully analyzed {len(results)} files")

    if not results:
        logger.error("No valid TextGrid files could be analyzed!")
        sys.exit(1)

    # Aggregate statistics
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE DURATION STATISTICS")
    logger.info("="*80)

    all_mean_durs_sec = [r['mean_dur_sec'] for r in results]
    all_std_durs_sec = [r['std_dur_sec'] for r in results]
    all_min_durs_sec = [r['min_dur_sec'] for r in results]
    all_max_durs_sec = [r['max_dur_sec'] for r in results]

    all_mean_durs_frames = [r['mean_dur_frames'] for r in results]
    all_std_durs_frames = [r['std_dur_frames'] for r in results]

    logger.info(f"\nAcross {len(results)} files:")
    logger.info(f"\nMean duration per phoneme (across files):")
    logger.info(f"  Average: {np.mean(all_mean_durs_sec):.4f}s ({np.mean(all_mean_durs_frames):.2f} frames)")
    logger.info(f"  Std:     {np.std(all_mean_durs_sec):.4f}s ({np.std(all_mean_durs_frames):.2f} frames)")
    logger.info(f"  Range:   [{np.min(all_mean_durs_sec):.4f}, {np.max(all_mean_durs_sec):.4f}]s")

    logger.info(f"\nWithin-file variance (std):")
    logger.info(f"  Average within-file std: {np.mean(all_std_durs_sec):.4f}s ({np.mean(all_std_durs_frames):.2f} frames)")
    logger.info(f"  Range: [{np.min(all_std_durs_sec):.4f}, {np.max(all_std_durs_sec):.4f}]s")

    logger.info(f"\nGlobal duration range:")
    logger.info(f"  Shortest phoneme: {np.min(all_min_durs_sec):.4f}s ({np.min(all_min_durs_sec)*22050/256:.2f} frames)")
    logger.info(f"  Longest phoneme:  {np.max(all_max_durs_sec):.4f}s ({np.max(all_max_durs_sec)*22050/256:.2f} frames)")

    # Check for uniform durations
    avg_within_std = np.mean(all_std_durs_frames)

    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS")
    logger.info("="*80)

    if avg_within_std < 1.0:
        logger.error("\n❌ CRITICAL ISSUE: Very low within-file variance!")
        logger.error(f"   Average std: {avg_within_std:.2f} frames")
        logger.error("   This suggests MFA alignments are nearly uniform")
        logger.error("   Expected: 2-5 frames std for realistic speech")
        logger.error("\n   Possible causes:")
        logger.error("   1. MFA dictionary doesn't match your phoneme set")
        logger.error("   2. MFA model is not suitable for this data")
        logger.error("   3. TextGrid files were post-processed/smoothed")
        logger.error("\n   RECOMMENDATION: Re-run MFA alignment or verify phoneme matching")
    elif avg_within_std < 2.0:
        logger.warning("\n⚠️  WARNING: Low within-file variance")
        logger.warning(f"   Average std: {avg_within_std:.2f} frames")
        logger.warning("   Expected: 2-5 frames std for realistic speech")
        logger.warning("   MFA alignments may not be capturing fine-grained timing")
    else:
        logger.info(f"\n✓ Good variance: Average within-file std = {avg_within_std:.2f} frames")
        logger.info("  MFA alignments appear to have realistic timing variation")

    # Analyze phoneme-specific durations
    analyze_phoneme_specific_durations(results)

    # Show example from one file
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE FROM FIRST FILE")
    logger.info("="*80)

    example = results[0]
    logger.info(f"\nFile: {example['filename']}")
    logger.info(f"Number of phonemes: {example['num_phonemes']}")
    logger.info(f"Mean duration: {example['mean_dur_sec']:.4f}s ({example['mean_dur_frames']:.2f} frames)")
    logger.info(f"Std deviation: {example['std_dur_sec']:.4f}s ({example['std_dur_frames']:.2f} frames)")
    logger.info(f"\nFirst 20 phonemes with durations:")
    for i in range(min(20, len(example['phonemes']))):
        phoneme = example['phonemes'][i]
        dur_sec = example['durations_sec'][i]
        dur_frames = example['durations_frames'][i]
        logger.info(f"  {phoneme:6s}: {dur_sec:.4f}s ({dur_frames:6.2f} frames)")

    # Check if durations look suspiciously uniform
    example_durs = example['durations_frames'][:20]
    if len(set([round(d, 1) for d in example_durs])) <= 3:
        logger.warning("\n⚠️  First 20 phonemes have very similar durations!")
        logger.warning("   This is suspicious for natural speech")

    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()
