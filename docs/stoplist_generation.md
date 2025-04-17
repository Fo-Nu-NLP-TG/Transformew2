# Stoplist Generation in TransformEw2

## What is a Stoplist?

A stoplist (also called a "stopword list") is a collection of common words that appear frequently in a language but carry little meaningful information on their own. Examples in English include words like "the", "and", "of", "in", etc.

## Why Do We Need Stoplists?

In machine translation, especially for low-resource languages like Ewe:

1. **Focus on Content Words**: By identifying stopwords, we can give more attention to content words that carry the main meaning.

2. **Reduce Noise**: Stopwords can sometimes introduce noise in the translation process.

3. **Improve Efficiency**: Filtering out very common words can make processing more efficient.

## TransformEw2's Dual-Method Approach

TransformEw2 uses a sophisticated dual-method approach to generate stoplists:

### 1. Regex-Based Method

This method uses regular expression patterns to identify common function words based on predefined patterns:

```python
# Example patterns for Ewe
patterns = [
    r'\b(la|na|be|ke|le|ɖe|kple|gake|eye)\b',  # Common function words
    r'\b(nye|wo|mi|mia|ye|ame)\b',  # Pronouns
    r'\b(ɖo|yi|va|le|nu)\b',  # Common verbs
]
```

**Advantages**:
- Works without requiring large amounts of text
- Can be tailored specifically for linguistic features of Ewe
- Captures known function words with high precision

### 2. Frequency-Based Method

This method identifies stopwords based on their frequency in the corpus:

```python
# Words appearing with frequency > threshold are considered stopwords
stopwords = {word for word, count in word_count.items() 
            if count/total_words > threshold}
```

**Advantages**:
- Automatically adapts to the specific corpus
- Can identify stopwords that might be missed by predefined patterns
- Works well for discovering corpus-specific common words

### 3. Combined Approach

TransformEw2 combines both methods to create a more comprehensive stoplist:

```python
# Generate stoplists using both methods
ewe_stopwords.update(regex_based_stoplist(ewe_corpus, 'ewe'))
ewe_stopwords.update(frequency_based_stoplist(ewe_corpus, 'ewe'))
```

**Advantages**:
- More comprehensive coverage than either method alone
- Balances linguistic knowledge with statistical patterns
- Particularly effective for low-resource languages like Ewe

## Example Stopwords in Ewe

Some common Ewe stopwords identified by our system include:

- `la` (the)
- `na` (to, for)
- `be` (that)
- `ke` (and)
- `le` (in, at)
- `ɖe` (on, at)
- `kple` (with)
- `eye` (and)

## How Stoplists Are Used in TransformEw2

Once generated, stoplists are used in several ways:

1. **Data Preprocessing**: Optionally filtering out stopwords during text cleaning
2. **Model Attention**: Helping the model focus on more meaningful words
3. **Evaluation**: Improving evaluation metrics by giving more weight to content words

## Testing Stoplist Generation

We've created comprehensive tests to verify our stoplist generation:

```python
def test_dual_method_approach(self):
    """Test that the dual-method approach combines results from both methods."""
    # Generate stoplists using both methods
    ewe_stopwords, english_stopwords = self.generator.generate_stoplists(
        self.ewe_corpus, self.english_corpus
    )
    
    # Check that regex-based stopwords are included
    regex_ewe = self.generator.regex_based_stoplist(self.ewe_corpus, 'ewe')
    
    # The final stoplist should be a superset of both methods
    self.assertTrue(regex_ewe.issubset(ewe_stopwords))
```

These tests ensure that our stoplist generation is working correctly and providing the expected benefits for Ewe-English translation.
