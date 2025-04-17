import unittest
import sys
import os
import pandas as pd
import re
import random
import numpy as np
from collections import Counter

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the modules we want to test
from data_processing.stoplist_generator import StoplistGenerator
from data_processing.data_augmentation import (
    augment_translation_data_ewe,
    smart_word_dropout,
    smart_word_swap,
    improved_simulate_back_translation,
    function_insertion,
    EWE_VERBS,
    EWE_FUNCTION_WORDS
)
from data_processing.data_cleaner import DataCleaner

class TestStoplistGenerator(unittest.TestCase):
    """Test suite for the enhanced stoplist generation in TransformEw2."""

    def setUp(self):
        """Set up test data."""
        self.generator = StoplistGenerator()

        # Sample Ewe corpus with known function words
        self.ewe_corpus = """
        me le tɔ me ɖu ƒu la kple ɖevi la
        ɖevi la ɖu nu nyuie eye wòyi suku
        ŋutsu la zɔ mɔ didi le ŋdi me eye wòva ɖo aƒe me le fiẽ
        nyɔnu la ƒle avɔ yeye na ɖevi la
        """

        # Sample English corpus with known stopwords
        self.english_corpus = """
        The child went to school and learned many things.
        She is reading a book about the history of Ghana.
        They have been working on this project for three weeks.
        We will visit the museum if the weather is good.
        """

    def test_regex_based_stoplist_ewe(self):
        """Test that regex-based stoplist detection works for Ewe."""
        stopwords = self.generator.regex_based_stoplist(self.ewe_corpus, 'ewe')

        # Check that common Ewe function words are detected
        # Only include words that are actually in the patterns
        expected_words = {'la', 'kple', 'eye', 'le', 'ɖo', 'va', 'yi'}
        detected_words = expected_words.intersection(stopwords)

        # At least 70% of expected words should be detected
        detection_rate = len(detected_words) / len(expected_words)
        self.assertGreaterEqual(detection_rate, 0.7,
                               f"Only detected {detected_words} out of {expected_words}")

    def test_regex_based_stoplist_english(self):
        """Test that regex-based stoplist detection works for English."""
        stopwords = self.generator.regex_based_stoplist(self.english_corpus, 'english')

        # Check that common English stopwords are detected
        expected_words = {'the', 'and', 'is', 'a', 'of', 'on', 'for', 'if', 'have', 'been'}
        detected_words = expected_words.intersection(stopwords)

        # At least 80% of expected words should be detected (English patterns are more reliable)
        detection_rate = len(detected_words) / len(expected_words)
        self.assertGreaterEqual(detection_rate, 0.8,
                               f"Only detected {detected_words} out of {expected_words}")

    def test_frequency_based_stoplist(self):
        """Test that frequency-based stoplist detection works."""
        # Create a corpus with repeated words to ensure they exceed the threshold
        repeated_corpus = "the the the the a a a a an an an is is is was was of of of in in"
        stopwords = self.generator.frequency_based_stoplist(repeated_corpus, 'english', threshold=0.05)

        # Words that appear frequently should be in the stoplist
        expected_frequent_words = {'the', 'a', 'an', 'is', 'was', 'of', 'in'}
        for word in expected_frequent_words:
            self.assertIn(word, stopwords, f"Frequent word '{word}' not detected as stopword")

    def test_dual_method_approach(self):
        """Test that the dual-method approach combines results from both methods."""
        # Generate stoplists using both methods
        ewe_stopwords, english_stopwords = self.generator.generate_stoplists(
            self.ewe_corpus, self.english_corpus
        )

        # Check that regex-based stopwords are included
        regex_ewe = self.generator.regex_based_stoplist(self.ewe_corpus, 'ewe')
        regex_english = self.generator.regex_based_stoplist(self.english_corpus, 'english')

        # Check that frequency-based stopwords are included
        freq_ewe = self.generator.frequency_based_stoplist(self.ewe_corpus, 'ewe')
        freq_english = self.generator.frequency_based_stoplist(self.english_corpus, 'english')

        # The final stoplist should be a superset of both methods
        self.assertTrue(regex_ewe.issubset(ewe_stopwords),
                       "Regex-based Ewe stopwords not fully included in final stoplist")
        self.assertTrue(regex_english.issubset(english_stopwords),
                       "Regex-based English stopwords not fully included in final stoplist")
        self.assertTrue(freq_ewe.issubset(ewe_stopwords),
                       "Frequency-based Ewe stopwords not fully included in final stoplist")
        self.assertTrue(freq_english.issubset(english_stopwords),
                       "Frequency-based English stopwords not fully included in final stoplist")


class TestDataAugmentationImprovements(unittest.TestCase):
    """Test suite for the enhanced data augmentation techniques in TransformEw2."""

    def setUp(self):
        """Set up test data."""
        # Fix random seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Create a small test dataset
        self.test_df = pd.DataFrame({
            'ewe': [
                'me le tɔ me ɖu ƒu',  # "I am swimming in the river"
                'ɖevi la ɖu nu nyuie',  # "The child ate well"
                'ŋutsu la zɔ mɔ didi',  # "The man walked a long distance"
                'nyɔnu la ƒle avɔ yeye',  # "The woman bought new cloth"
            ],
            'english': [
                'I am swimming in the river',
                'The child ate well',
                'The man walked a long distance',
                'The woman bought new cloth',
            ]
        })

        # Sample Ewe text with known verbs for specific tests
        self.verb_text = 'ɖevi la ɖu nu nyuie kple lɔ̃ nɔvi'  # Contains verb 'ɖu' (eat)
        self.long_text = 'ŋutsu la zɔ mɔ didi le ŋdi me eye wòva ɖo aƒe me le fiẽ'

    def test_verb_preservation_in_augmentation(self):
        """Test that verbs are preserved in augmentation techniques."""
        # Apply all augmentation techniques
        augmented_df = augment_translation_data_ewe(
            self.test_df,
            src_col='ewe',
            tgt_col='english',
            techniques=['word_dropout', 'word_swap', 'back_translation', 'function_insertion']
        )

        # Check each augmented sentence for verb preservation
        for idx, row in augmented_df.iterrows():
            ewe_text = row['ewe']
            words = ewe_text.split()

            # Count verbs in the augmented text
            verb_count = sum(1 for word in words if word in EWE_VERBS)

            # There should be at least one verb in each sentence (if the original had verbs)
            original_verbs = [w for w in self.test_df['ewe'].iloc[0].split() if w in EWE_VERBS]
            if original_verbs:
                self.assertGreater(verb_count, 0,
                                  f"No verbs found in augmented text: {ewe_text}")

    def test_function_word_insertion(self):
        """Test the function word insertion technique (new in TransformEw2)."""
        # Apply function word insertion
        result = function_insertion(self.verb_text, prob=0.2)

        # The result should be longer than the original
        self.assertGreater(len(result.split()), len(self.verb_text.split()))

        # At least one function word should be inserted
        original_words = set(self.verb_text.split())
        result_words = set(result.split())
        new_words = result_words - original_words

        function_words_added = any(word in EWE_FUNCTION_WORDS for word in new_words)
        self.assertTrue(function_words_added,
                       f"No function words added. New words: {new_words}")

    def test_improved_back_translation_simulation(self):
        """Test the improved back translation simulation (enhanced in TransformEw2)."""
        result = improved_simulate_back_translation(self.long_text)

        # Result should be different from the original
        self.assertNotEqual(result, self.long_text)

        # Result should preserve most verbs
        original_verbs = [w for w in self.long_text.split() if w in EWE_VERBS]
        result_verbs = [w for w in result.split() if w in EWE_VERBS]

        # At least 70% of verbs should be preserved
        if original_verbs:
            preservation_rate = len(result_verbs) / len(original_verbs)
            self.assertGreaterEqual(preservation_rate, 0.7,
                                   f"Only preserved {result_verbs} out of {original_verbs}")

    def test_multiple_augmentation_techniques(self):
        """Test that multiple augmentation techniques can be applied together."""
        original_size = len(self.test_df)

        # Apply all techniques
        augmented_df = augment_translation_data_ewe(
            self.test_df,
            src_col='ewe',
            tgt_col='english',
            techniques=['word_dropout', 'word_swap', 'back_translation', 'function_insertion']
        )

        # Should have more rows than the original (one for each technique plus original)
        expected_min_size = original_size * 2  # At least original + some augmentations
        self.assertGreaterEqual(len(augmented_df), expected_min_size,
                               f"Expected at least {expected_min_size} rows, got {len(augmented_df)}")

        # Check that each technique produced different results
        unique_texts = augmented_df['ewe'].nunique()
        self.assertGreater(unique_texts, original_size,
                          f"Expected more than {original_size} unique texts, got {unique_texts}")


class TestDataCleanerImprovements(unittest.TestCase):
    """Test suite for the improved data cleaning in TransformEw2."""

    def setUp(self):
        """Set up test data."""
        self.cleaner = DataCleaner(data_dir="./tests/test_data")  # Use a test directory

        # Create test directory if it doesn't exist
        os.makedirs("./tests/test_data", exist_ok=True)

        # Sample data with various issues
        self.test_data = pd.DataFrame({
            'Ewe': [
                'me le <b>tɔ</b> me ɖu ƒu',  # HTML tags
                'ɖevi   la  ɖu  nu   nyuie',  # Extra whitespace
                'ŋutsu la zɔ mɔ didi!!!',  # Special characters
                '',  # Empty string
                'a',  # Too short
                'nyɔnu la ƒle avɔ yeye ' + 'na ɖevi ' * 20,  # Too long
            ],
            'English': [
                'I am <i>swimming</i> in the river',
                'The   child  ate  well',
                'The man walked a long distance!!!',
                'This has no Ewe translation',
                'b',
                'The woman bought new cloth ' + 'for the child ' * 20,
            ]
        })

        # Save test data
        self.test_data.to_csv("./tests/test_data/ewe_english.csv", index=False)

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if os.path.exists("./tests/test_data"):
            shutil.rmtree("./tests/test_data")

    def test_html_tag_removal(self):
        """Test that HTML tags are removed."""
        cleaned = self.cleaner.clean_text('me le <b>tɔ</b> me ɖu ƒu')
        self.assertNotIn('<b>', cleaned)
        self.assertNotIn('</b>', cleaned)
        self.assertEqual(cleaned, 'me le tɔ me ɖu ƒu')

    def test_whitespace_normalization(self):
        """Test that extra whitespace is normalized."""
        cleaned = self.cleaner.clean_text('ɖevi   la  ɖu  nu   nyuie')
        self.assertNotIn('  ', cleaned)
        self.assertEqual(cleaned, 'ɖevi la ɖu nu nyuie')

    def test_special_character_handling(self):
        """Test that special characters are handled appropriately."""
        cleaned = self.cleaner.clean_text('ŋutsu la zɔ mɔ didi@#$%^&*')
        self.assertNotIn('@#$%^&*', cleaned)
        # But it should preserve Ewe-specific characters
        self.assertIn('ŋ', cleaned)

    def test_length_filtering(self):
        """Test that sentences are filtered by length."""
        filtered = self.cleaner.filter_by_length(
            self.test_data, 'Ewe', 'English', min_len=2, max_len=10
        )

        # Should exclude empty strings, too short, and too long sentences
        expected_rows = 3  # Only the first 3 rows should remain
        self.assertEqual(len(filtered), expected_rows)

        # Check that specific problematic rows are excluded
        self.assertNotIn('', filtered['Ewe'].values)
        self.assertNotIn('a', filtered['Ewe'].values)

    def test_full_preprocessing_pipeline(self):
        """Test the full preprocessing pipeline."""
        # Process the test dataset
        processed_df = self.cleaner.preprocess_dataset(
            "ewe_english.csv", "Ewe", "English", "ee", "en"
        )

        # Check that the processed data has been cleaned
        self.assertIsNotNone(processed_df)
        self.assertLess(len(processed_df), len(self.test_data))

        # Check that corpus files were created
        self.assertTrue(os.path.exists("./tests/test_data/ee_corpus.txt"))
        self.assertTrue(os.path.exists("./tests/test_data/en_corpus.txt"))

        # Verify that cleaned data was saved
        self.assertTrue(os.path.exists("./tests/test_data/clean_ewe_english.csv"))


if __name__ == '__main__':
    unittest.main()
