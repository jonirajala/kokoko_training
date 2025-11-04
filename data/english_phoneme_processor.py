#!/usr/bin/env python3
"""
English Phoneme Processor using g2p_en (ARPA phonemes)
Provides grapheme-to-phoneme conversion for English TTS training

SWITCHED FROM MISAKI (IPA) TO G2P_EN (ARPA) FOR 100% MFA ALIGNMENT MATCH
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EnglishPhonemeProcessor:
    """
    English phoneme processor using g2p_en (ARPA phonemes).

    This matches MFA's english_us_arpa model perfectly for 100% alignment usage.
    """

    def __init__(self, variant: str = 'en-us'):
        """
        Initialize the English phoneme processor with ARPA phonemes.

        Args:
            variant: Language variant (kept for compatibility, g2p_en is US English)
        """
        self.variant = variant

        # Store g2p as None - will be lazy loaded
        # This avoids pickle issues with multiprocessing DataLoader
        self._g2p = None
        self.use_g2p_en = True

        # Download required NLTK data (but don't create G2p instance yet)
        try:
            import nltk
            # Download required NLTK data (silently if already present)
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            except LookupError:
                logger.info("Downloading required NLTK data...")
                nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                nltk.download('cmudict', quiet=True)
            logger.info(f"Initialized English phoneme processor with g2p_en (ARPA phonemes)")
        except ImportError as e:
            logger.error(
                f"g2p_en not found ({e}). Install with: pip install g2p_en\n"
            )
            raise ImportError("g2p_en is required for ARPA phoneme processing")

        # Build ARPA phoneme vocabulary
        self.phoneme_to_id = self._build_vocab()
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
        logger.info(f"Vocabulary size: {len(self.phoneme_to_id)} ARPA phonemes")

    @property
    def g2p(self):
        """Lazy-load G2p instance to avoid pickle issues with multiprocessing"""
        if self._g2p is None:
            from g2p_en import G2p
            self._g2p = G2p()
        return self._g2p

    def _build_vocab(self) -> Dict[str, int]:
        """
        Build ARPA phoneme vocabulary matching MFA's english_us_arpa model.

        ARPA phoneme set (CMU pronouncing dictionary):
        - Vowels: AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW
        - Consonants: B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH, T, TH, V, W, Y, Z, ZH
        - Stress markers: 0, 1, 2 (added as suffix to vowels)

        Returns:
            Dictionary mapping phonemes to integer IDs
        """
        # Start with special tokens
        vocab = {
            '<pad>': 0,   # Padding token
            '<unk>': 1,   # Unknown phoneme
            ' ': 2,       # Space/word boundary
        }

        # ARPA vowels (no stress markers - we'll handle stressed variants separately)
        vowels = [
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
            'EH', 'ER', 'EY', 'IH', 'IY', 'OW',
            'OY', 'UH', 'UW'
        ]

        # ARPA consonants
        consonants = [
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH',
            'JH', 'K', 'L', 'M', 'N', 'NG', 'P',
            'R', 'S', 'SH', 'T', 'TH', 'V', 'W',
            'Y', 'Z', 'ZH'
        ]

        # Add vowels with stress markers (0, 1, 2)
        vowel_variants = []
        for vowel in vowels:
            vowel_variants.append(vowel)  # Base form
            for stress in ['0', '1', '2']:
                vowel_variants.append(vowel + stress)

        # Add punctuation that g2p_en outputs
        punctuation = ['.', ',', '!', '?', ':', ';', '-', '"', "'"]

        # Combine all phonemes
        all_phonemes = vowel_variants + consonants + punctuation

        # Add to vocabulary
        for phoneme in all_phonemes:
            if phoneme not in vocab:
                vocab[phoneme] = len(vocab)

        return vocab

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.phoneme_to_id)

    def process_text(self, text: str) -> List[List[str]]:
        """
        Process text and return phoneme sequence in the format expected by training.
        This is an alias for compatibility with the training pipeline.

        Args:
            text: Input text string

        Returns:
            List containing a single list of phonemes (word-level structure)
        """
        phonemes = self.text_to_phonemes(text)
        # Return in the format expected by training: [[phonemes]]
        return [phonemes] if phonemes else [[]]

    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to ARPA phoneme sequence using g2p_en.

        Args:
            text: Input text string

        Returns:
            List of ARPA phoneme strings (e.g., ['HH', 'EH1', 'L', 'OW0'])
        """
        if not text or not text.strip():
            return []

        try:
            # g2p_en converts text to phonemes
            # Returns list like: ['HH', 'AH0', 'L', 'OW1', ' ', 'W', 'ER1', 'L', 'D']
            phonemes = self.g2p(text)

            # Filter out empty strings and clean
            phonemes = [p for p in phonemes if p and p.strip()]

            return phonemes

        except Exception as e:
            logger.error(f"Error in g2p_en conversion: {e}")
            logger.warning("Returning empty phoneme list")
            return []

    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert text to phoneme indices.

        Args:
            text: Input text string

        Returns:
            List of phoneme indices
        """
        phonemes = self.text_to_phonemes(text)
        indices = []

        for phoneme in phonemes:
            if phoneme in self.phoneme_to_id:
                indices.append(self.phoneme_to_id[phoneme])
            else:
                # Unknown phoneme
                logger.warning(f"Unknown phoneme: {phoneme}, using <unk>")
                indices.append(self.phoneme_to_id['<unk>'])

        return indices

    def indices_to_phonemes(self, indices: List[int]) -> List[str]:
        """
        Convert phoneme indices back to phoneme strings.

        Args:
            indices: List of phoneme indices

        Returns:
            List of phoneme strings
        """
        phonemes = []
        for idx in indices:
            if idx in self.id_to_phoneme:
                phonemes.append(self.id_to_phoneme[idx])
            else:
                phonemes.append('<unk>')

        return phonemes

    def phonemes_to_text(self, phonemes: List[str]) -> str:
        """
        Convert phonemes back to approximate text (not perfect, for debugging).

        Args:
            phonemes: List of phoneme strings

        Returns:
            Approximate text string
        """
        # This is approximate - ARPA -> text is lossy
        return ' '.join(phonemes)

    def to_dict(self) -> dict:
        """
        Convert processor to dictionary for serialization.

        Returns:
            Dictionary containing processor state
        """
        return {
            'variant': self.variant,
            'phoneme_to_id': self.phoneme_to_id,
            'id_to_phoneme': self.id_to_phoneme,
            'use_g2p_en': self.use_g2p_en,
            'processor_type': 'g2p_en_arpa'
        }

    @classmethod
    def from_dict(cls, state_dict: dict) -> 'EnglishPhonemeProcessor':
        """
        Create processor from dictionary.

        Args:
            state_dict: Dictionary containing processor state

        Returns:
            EnglishPhonemeProcessor instance
        """
        # Create new processor instance
        processor = cls(variant=state_dict.get('variant', 'en-us'))

        # Override with saved vocabulary if available
        if 'phoneme_to_id' in state_dict:
            processor.phoneme_to_id = state_dict['phoneme_to_id']
            processor.id_to_phoneme = state_dict['id_to_phoneme']

        return processor


if __name__ == "__main__":
    # Test the processor
    processor = EnglishPhonemeProcessor()

    test_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Printing, in the only sense with which we are at present concerned"
    ]

    print("\n" + "="*80)
    print("ARPA Phoneme Processor Test")
    print("="*80)

    for text in test_texts:
        print(f"\nText: {text}")
        phonemes = processor.text_to_phonemes(text)
        print(f"Phonemes ({len(phonemes)}): {' '.join(phonemes)}")
        indices = processor.text_to_indices(text)
        print(f"Indices ({len(indices)}): {indices[:20]}...")

    print(f"\nVocabulary size: {processor.get_vocab_size()}")
    print("\n" + "="*80)
