# Testing Approach in TransformEw2

## Why Testing Matters

Testing is crucial for ensuring that our machine translation system works correctly and reliably. For TransformEw2, we've developed a comprehensive testing strategy that focuses on:

1. **Verifying Improvements**: Confirming that the enhancements made in TransformEw2 actually work as intended
2. **Preventing Regressions**: Ensuring that new changes don't break existing functionality
3. **Documenting Behavior**: Tests serve as executable documentation of how components should work
4. **Quality Assurance**: Catching bugs and issues before they affect translation quality

## Test Suite Organization

Our tests are organized into several categories, each focusing on a specific component of TransformEw2:

### 1. Stoplist Generation Tests

These tests verify that our stoplist generation correctly identifies function words in both Ewe and English:

```python
class TestStoplistGenerator(unittest.TestCase):
    def test_regex_based_stoplist_ewe(self):
        """Test that regex-based stoplist detection works for Ewe."""
        stopwords = self.generator.regex_based_stoplist(self.ewe_corpus, 'ewe')
        
        # Check that common Ewe function words are detected
        expected_words = {'la', 'kple', 'eye', 'le', 'me', 'ɖu', 'ɖo', 'va', 'yi'}
        detected_words = expected_words.intersection(stopwords)
        
        # At least 70% of expected words should be detected
        detection_rate = len(detected_words) / len(expected_words)
        self.assertGreaterEqual(detection_rate, 0.7)
```

### 2. Data Augmentation Tests

These tests verify that our data augmentation techniques create valid variations while preserving meaning:

```python
class TestDataAugmentationImprovements(unittest.TestCase):
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
                self.assertGreater(verb_count, 0)
```

### 3. Data Cleaning Tests

These tests verify that our data cleaning pipeline correctly handles various text issues:

```python
class TestDataCleanerImprovements(unittest.TestCase):
    def test_html_tag_removal(self):
        """Test that HTML tags are removed."""
        cleaned = self.cleaner.clean_text('me le <b>tɔ</b> me ɖu ƒu')
        self.assertNotIn('<b>', cleaned)
        self.assertNotIn('</b>', cleaned)
        self.assertEqual(cleaned, 'me le tɔ me ɖu ƒu')
```

## Test Data

We use several types of test data:

1. **Synthetic Examples**: Hand-crafted examples designed to test specific edge cases
2. **Real-World Samples**: Small subsets of actual Ewe-English parallel text
3. **Problematic Cases**: Examples that caused issues in TransformEw1

## Running the Tests

To run all tests for TransformEw2:

```bash
cd tests
python -m unittest discover
```

To run tests for a specific component:

```bash
cd tests
python -m unittest test_transformew2_improvements.py
```

## Continuous Testing

We recommend running tests:

1. After making any changes to the codebase
2. Before processing new datasets
3. Before training new models
4. As part of any continuous integration pipeline

## Test Coverage

Our test suite covers:

1. **Stoplist Generation**:
   - Regex-based detection for both Ewe and English
   - Frequency-based detection
   - Combined dual-method approach

2. **Data Augmentation**:
   - Verb preservation during augmentation
   - Function word insertion
   - Simulated back-translation
   - Multiple augmentation techniques

3. **Data Cleaning**:
   - HTML tag removal
   - Whitespace normalization
   - Special character handling
   - Length-based filtering
   - Full preprocessing pipeline

## Extending the Tests

When adding new features to TransformEw2, we recommend:

1. Writing tests before implementing the feature (Test-Driven Development)
2. Ensuring tests cover both normal operation and edge cases
3. Adding tests for any bugs discovered during development
4. Updating existing tests when changing behavior

By maintaining a comprehensive test suite, we ensure that TransformEw2 continues to provide high-quality Ewe-English translation even as the codebase evolves.
