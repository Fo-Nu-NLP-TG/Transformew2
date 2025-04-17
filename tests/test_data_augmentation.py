import unittest
import pandas as pd
import sys
import os
import random
import numpy as np

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_augmentation import (
    augment_translation_data_ewe,
    smart_word_dropout,
    smart_word_swap,
    improved_simulate_back_translation,
    function_insertion,
    EWE_VERBS,
    EWE_FUNCTION_WORDS
)

class TestDataAugmentation(unittest.TestCase):
    """Test suite for data augmentation techniques used in Ewe-English translation."""
    
    def setUp(self):
        """Set up test data that will be used across multiple tests."""
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
    
    def test_smart_word_dropout_preserves_verbs(self):
        """Test that smart_word_dropout preserves Ewe verbs."""
        # Apply dropout with high probability to ensure some words are dropped
        result = smart_word_dropout(self.verb_text, prob=0.9)
        
        # Check that the verb 'ɖu' is still in the result
        self.assertIn('ɖu', result.split())
        
        # The result should be shorter than the original (some words dropped)
        self.assertLess(len(result.split()), len(self.verb_text.split()))
    
    def test_smart_word_dropout_respects_min_words(self):
        """Test that smart_word_dropout respects the minimum word count."""
        short_text = 'ɖevi ɖu nu'  # 3 words
        
        # With min_words=3, no words should be dropped
        result = smart_word_dropout(short_text, prob=1.0, min_words=3)
        self.assertEqual(len(result.split()), 3)
        
        # With min_words=2, some words might be dropped but at least 2 remain
        result = smart_word_dropout(short_text, prob=1.0, min_words=2)
        self.assertGreaterEqual(len(result.split()), 2)
    
    def test_smart_word_swap_preserves_verbs(self):
        """Test that smart_word_swap doesn't swap verbs."""
        # Set a high probability to ensure swaps happen
        result = smart_word_swap(self.verb_text, prob=1.0)
        
        # Get the positions of verbs in original and result
        original_words = self.verb_text.split()
        result_words = result.split()
        
        verb_positions_original = [i for i, word in enumerate(original_words) if word in EWE_VERBS]
        verb_positions_result = [i for i, word in enumerate(result_words) if word in EWE_VERBS]
        
        # Verb positions should be the same
        self.assertEqual(verb_positions_original, verb_positions_result)
        
        # The words themselves should be preserved
        for pos in verb_positions_original:
            self.assertEqual(original_words[pos], result_words[pos])
    
    def test_improved_simulate_back_translation(self):
        """Test that improved_simulate_back_translation produces valid output."""
        result = improved_simulate_back_translation(self.long_text)
        
        # Result should not be empty
        self.assertTrue(result.strip())
        
        # Result should be a modified version of the original
        self.assertNotEqual(result, self.long_text)
        
        # Result should contain most of the verbs from the original
        original_verbs = [w for w in self.long_text.split() if w in EWE_VERBS]
        result_verbs = [w for w in result.split() if w in EWE_VERBS]
        
        # At least 80% of verbs should be preserved
        self.assertGreaterEqual(len(result_verbs), len(original_verbs) * 0.8)
    
    def test_function_insertion(self):
        """Test that function_insertion adds Ewe function words."""
        original_word_count = len(self.verb_text.split())
        result = function_insertion(self.verb_text, prob=0.2)
        
        # Result should have more words than the original
        self.assertGreater(len(result.split()), original_word_count)
        
        # At least one of the added words should be from EWE_FUNCTION_WORDS
        added_words = set(result.split()) - set(self.verb_text.split())
        function_words_added = any(word in EWE_FUNCTION_WORDS for word in added_words)
        self.assertTrue(function_words_added)
    
    def test_augment_translation_data_ewe_increases_data(self):
        """Test that augment_translation_data_ewe increases the dataset size."""
        original_size = len(self.test_df)
        augmented_df = augment_translation_data_ewe(
            self.test_df, 
            src_col='ewe', 
            tgt_col='english',
            techniques=['word_dropout', 'word_swap']
        )
        
        # Augmented dataset should be larger than original
        self.assertGreater(len(augmented_df), original_size)
    
    def test_augment_translation_data_ewe_preserves_target(self):
        """Test that augment_translation_data_ewe preserves target language text."""
        augmented_df = augment_translation_data_ewe(
            self.test_df, 
            src_col='ewe', 
            tgt_col='english',
            techniques=['word_dropout']
        )
        
        # All original English sentences should be in the augmented dataset
        original_english = set(self.test_df['english'])
        augmented_english = set(augmented_df['english'])
        
        self.assertTrue(original_english.issubset(augmented_english))
    
    def test_augment_translation_data_ewe_filters_low_quality(self):
        """Test that augment_translation_data_ewe filters out low-quality augmentations."""
        # Create a dataframe with one very short sentence
        short_df = pd.DataFrame({
            'ewe': ['me le'],  # Just "I am"
            'english': ['I am']
        })
        
        augmented_df = augment_translation_data_ewe(
            short_df, 
            src_col='ewe', 
            tgt_col='english',
            techniques=['word_dropout', 'word_swap', 'function_insertion']
        )
        
        # The augmented dataset should only contain the original row
        # since augmentations would be filtered out for being too short
        self.assertEqual(len(augmented_df), 1)
    
    def test_empty_input_handling(self):
        """Test that all functions handle empty input gracefully."""
        empty_text = ""
        
        # All functions should return empty string for empty input
        self.assertEqual(smart_word_dropout(empty_text), empty_text)
        self.assertEqual(smart_word_swap(empty_text), empty_text)
        self.assertEqual(improved_simulate_back_translation(empty_text), empty_text)
        self.assertEqual(function_insertion(empty_text), empty_text)

if __name__ == '__main__':
    unittest.main()
