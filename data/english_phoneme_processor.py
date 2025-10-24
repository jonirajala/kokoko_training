#!/usr/bin/env python3
"""
English Phoneme Processor using Misaki G2P
Provides grapheme-to-phoneme conversion for English TTS training
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnglishPhonemeProcessor:
    """
    English phoneme processor using Misaki G2P engine.

    This processor converts English text to phoneme sequences using the Misaki
    G2P library, which is designed for the Kokoro TTS architecture.
    """

    def __init__(self, variant: str = 'en-us'):
        """
        Initialize the English phoneme processor.

        Args:
            variant: Language variant ('en-us' or 'en-gb')
        """
        self.variant = variant

        # Try to import Misaki
        try:
            from misaki.en import G2P
            british = (variant == 'en-gb')
            # Use trf=False to avoid transformer dependency issues
            self.g2p = G2P(trf=False, british=british)
            self.use_misaki = True
            logger.info(f"Initialized English phoneme processor with Misaki ({variant})")
        except ImportError as e:
            logger.warning(
                f"Misaki not found ({e}). Install with: pip install 'misaki[en]'\n"
                "Falling back to basic phoneme mapping."
            )
            self.use_misaki = False
            self.g2p = None

        # Build phoneme vocabulary
        self.phoneme_to_id = self._build_vocab()
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}

        logger.info(f"Vocabulary size: {len(self.phoneme_to_id)} phonemes")

    def _build_vocab(self) -> Dict[str, int]:
        """
        Build phoneme vocabulary mapping.

        Based on Misaki's actual US_VOCAB phoneme set.
        See: misaki/en.py US_VOCAB

        Returns:
            Dictionary mapping phonemes to integer IDs
        """
        # Start with special tokens
        vocab = {
            '<pad>': 0,   # Padding token
            '<unk>': 1,   # Unknown phoneme
            ' ': 2,       # Space/word boundary
        }

        # Misaki US_VOCAB phoneme set (sorted for consistency)
        # US_VOCAB = frozenset('AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ')
        misaki_phonemes = [
            # Capital letters (diphthongs in Misaki notation)
            'A',   # /aɪ/ sound (as in "ride")
            'I',   # /aɪ/ alternative
            'O',   # /oʊ/ sound (as in "go")
            'W',   # /aʊ/ sound (as in "now")
            'Y',   # /ɔɪ/ sound (as in "boy")

            # Consonants
            'b', 'd', 'f', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 's', 't', 'v', 'w', 'z',
            'ɡ',   # voiced velar stop
            'ŋ',   # velar nasal (sing)
            'ð',   # voiced dental fricative (this)
            'θ',   # voiceless dental fricative (thin)
            'ʃ',   # voiceless postalveolar fricative (ship)
            'ʒ',   # voiced postalveolar fricative (measure)
            'ʧ',   # voiceless postalveolar affricate (church)
            'ʤ',   # voiced postalveolar affricate (judge)
            'ɹ',   # alveolar approximant (AmE r)
            'ɾ',   # alveolar flap/tap (AmE "butter")
            'T',   # Alternative notation for flap/tap

            # Vowels
            'i',   # close front unrounded (beat)
            'u',   # close back rounded (boot)
            'æ',   # near-open front unrounded (bat)
            'ɑ',   # open back unrounded (father)
            'ɔ',   # open-mid back rounded (thought)
            'ə',   # mid central (schwa - about)
            'ɛ',   # open-mid front unrounded (bet)
            'ɜ',   # open-mid central unrounded (bird)
            'ɪ',   # near-close near-front unrounded (bit)
            'ʊ',   # near-close near-back rounded (put)
            'ʌ',   # open-mid back unrounded (but)
            'ᵊ',   # reduced schwa
            'ᵻ',   # reduced close central unrounded
            'ɐ',   # near-open central vowel (schwa-like, used for unstressed "a")

            # Stress markers
            'ˈ',   # primary stress
            'ˌ',   # secondary stress

            # Special
            'ʔ',   # glottal stop
        ]

        # Add common punctuation that may appear in Misaki output
        punctuation = [
            '.',   # period
            ',',   # comma
            '!',   # exclamation
            '?',   # question
            '-',   # dash/hyphen
            '—',   # em dash (Misaki uses this)
            ':',   # colon
            ';',   # semicolon
            '"',   # quote
            "'",   # apostrophe
            '(',   # left paren
            ')',   # right paren
        ]

        # Combine Misaki phonemes and punctuation
        all_phonemes = misaki_phonemes + punctuation

        # Add to vocabulary (de-duplicate in case of overlap)
        for phoneme in all_phonemes:
            if phoneme not in vocab:
                vocab[phoneme] = len(vocab)

        return vocab

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
        Convert text to phoneme sequence using Misaki.

        Args:
            text: Input text string

        Returns:
            List of phoneme strings
        """
        if not text or not text.strip():
            return []

        if self.use_misaki:
            try:
                # Use Misaki G2P for conversion
                # G2P returns (phoneme_string, tokens)
                ipa_text, tokens = self.g2p(text)

                # Parse IPA string into individual phonemes
                phonemes = self._parse_ipa(ipa_text)

                return phonemes
            except Exception as e:
                logger.error(f"Error in Misaki G2P conversion: {e}")
                logger.warning("Falling back to character-level processing")
                return self._fallback_processing(text)
        else:
            # Fallback to basic processing
            return self._fallback_processing(text)

    def _parse_ipa(self, ipa_text: str) -> List[str]:
        """
        Parse IPA string into individual phonemes.

        Handles multi-character phonemes (like 'tʃ', 'dʒ', 'aɪ', etc.)

        Args:
            ipa_text: IPA string from Misaki

        Returns:
            List of individual phonemes
        """
        phonemes = []
        i = 0

        while i < len(ipa_text):
            # Try to match multi-character phonemes first
            matched = False

            # Check for 3-character phonemes
            if i + 2 < len(ipa_text):
                three_char = ipa_text[i:i+3]
                if three_char in self.phoneme_to_id:
                    phonemes.append(three_char)
                    i += 3
                    matched = True

            # Check for 2-character phonemes
            if not matched and i + 1 < len(ipa_text):
                two_char = ipa_text[i:i+2]
                if two_char in self.phoneme_to_id:
                    phonemes.append(two_char)
                    i += 2
                    matched = True

            # Check for single character
            if not matched:
                one_char = ipa_text[i]
                if one_char in self.phoneme_to_id:
                    phonemes.append(one_char)
                else:
                    # Unknown phoneme - log warning and use <unk>
                    if one_char not in [' ', '\n', '\t']:
                        logger.debug(f"Unknown phoneme: '{one_char}' (U+{ord(one_char):04X})")
                    if one_char.isspace():
                        phonemes.append(' ')
                    else:
                        phonemes.append('<unk>')
                i += 1

        return phonemes

    def _fallback_processing(self, text: str) -> List[str]:
        """
        Fallback text processing when Misaki is not available.

        This is a very basic character-level processing and should only be used
        for testing. For production use, install Misaki.

        Args:
            text: Input text

        Returns:
            List of characters (not true phonemes)
        """
        logger.warning("Using fallback character-level processing - not true phonemes!")

        # Convert to lowercase and split into characters
        text = text.lower()
        chars = []

        for char in text:
            if char.isalpha():
                chars.append(char)
            elif char.isspace():
                chars.append(' ')
            elif char in '.,!?-:;\'"':
                chars.append(char)

        return chars

    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert text to phoneme indices.

        Args:
            text: Input text string

        Returns:
            List of phoneme indices
        """
        phonemes = self.text_to_phonemes(text)
        indices = [
            self.phoneme_to_id.get(p, self.phoneme_to_id['<unk>'])
            for p in phonemes
        ]
        return indices

    def indices_to_phonemes(self, indices: List[int]) -> List[str]:
        """
        Convert phoneme indices back to phoneme strings.

        Args:
            indices: List of phoneme indices

        Returns:
            List of phoneme strings
        """
        phonemes = [
            self.id_to_phoneme.get(idx, '<unk>')
            for idx in indices
        ]
        return phonemes

    def indices_to_text(self, indices: List[int]) -> str:
        """
        Convert phoneme indices back to a string.

        Args:
            indices: List of phoneme indices

        Returns:
            Phoneme string
        """
        phonemes = self.indices_to_phonemes(indices)
        return ''.join(phonemes)

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.phoneme_to_id)

    def to_dict(self) -> Dict:
        """
        Serialize processor for saving.

        Returns:
            Dictionary containing processor state
        """
        return {
            'variant': self.variant,
            'phoneme_to_id': self.phoneme_to_id,
            'use_misaki': self.use_misaki,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EnglishPhonemeProcessor':
        """
        Deserialize processor from saved data.

        Args:
            data: Saved processor state

        Returns:
            Restored processor instance
        """
        processor = cls(variant=data.get('variant', 'en-us'))
        processor.phoneme_to_id = data['phoneme_to_id']
        processor.id_to_phoneme = {v: k for k, v in processor.phoneme_to_id.items()}
        return processor

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"EnglishPhonemeProcessor(variant='{self.variant}', "
            f"vocab_size={self.get_vocab_size()}, "
            f"use_misaki={self.use_misaki})"
        )


def test_processor():
    """Test the English phoneme processor"""
    processor = EnglishPhonemeProcessor('en-us')

    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "Text to speech synthesis.",
    ]

    print(f"\n{processor}")
    print(f"Vocabulary size: {processor.get_vocab_size()}\n")

    for text in test_texts:
        phonemes = processor.text_to_phonemes(text)
        indices = processor.text_to_indices(text)

        print(f"Text: {text}")
        print(f"Phonemes: {phonemes}")
        print(f"Indices: {indices}")
        print(f"Length: {len(phonemes)} phonemes\n")


if __name__ == "__main__":
    test_processor()
