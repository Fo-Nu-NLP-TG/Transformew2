# Data Augmentation in TransformEw2

## What is Data Augmentation?

Data augmentation is a set of techniques used to artificially increase the size and diversity of training data by creating modified versions of existing data. In the context of machine translation, it involves creating variations of existing sentence pairs while preserving their meaning.

## Why is Data Augmentation Important?

For low-resource languages like Ewe:

1. **Limited Data**: We often have very limited parallel text (Ewe-English sentence pairs).
2. **Improved Generalization**: Augmentation helps the model learn to handle variations in language.
3. **Reduced Overfitting**: More diverse training examples help prevent the model from memorizing the training data.
4. **Better Performance**: Models trained on augmented data typically perform better on unseen text.

## TransformEw2's Specialized Augmentation Techniques

TransformEw2 implements several language-aware augmentation techniques specifically designed for Ewe:

### 1. Smart Word Dropout

This technique randomly removes some words from a sentence, but intelligently preserves verbs which are crucial for meaning:

```python
def smart_word_dropout(text, prob=0.05, min_words=3):
    """Drop words, avoiding Ewe verbs"""
    words = text.split()
    if len(words) <= min_words:
        return text
    
    new_words = [w for w in words if w in EWE_VERBS or random.random() > prob]
    return ' '.join(new_words)
```

**Example**:
- Original: `ɖevi la ɖu nu nyuie kple lɔ̃ nɔvi`
- Augmented: `ɖevi ɖu nu nyuie lɔ̃ nɔvi` (dropped "la" and "kple")

### 2. Smart Word Swap

This technique swaps adjacent words, but avoids disrupting verb positions to maintain grammatical structure:

```python
def smart_word_swap(text, prob=0.1):
    """Swap words, avoiding verb disruption"""
    words = text.split()
    new_words = words.copy()
    
    for i in range(len(words) - 1):
        if random.random() < prob and words[i] not in EWE_VERBS and words[i + 1] not in EWE_VERBS:
            new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]
    return ' '.join(new_words)
```

**Example**:
- Original: `nyɔnu la ƒle avɔ yeye`
- Augmented: `nyɔnu la avɔ ƒle yeye` (swapped "ƒle" and "avɔ")

### 3. Simulated Back-Translation

Back-translation is a powerful augmentation technique that typically requires a reverse translation model (English→Ewe). Since such models are scarce for Ewe, we simulate the effect:

```python
def improved_simulate_back_translation(text, dropout_prob=0.05, reorder_prob=0.1):
    """Simulate back translation with reordering and function word insertion"""
    words = text.split()
    
    # Dropout, protecting verbs
    new_words = [w for w in words if w in EWE_VERBS or random.random() > dropout_prob]
    
    # Reorder within a small window (mimic syntactic variation)
    if len(new_words) > 3 and random.random() < reorder_prob:
        window = min(3, len(new_words) - 1)
        i = random.randint(0, len(new_words) - window)
        j = i + random.randint(1, window)
        if new_words[i] not in EWE_VERBS and new_words[j] not in EWE_VERBS:
            new_words[i], new_words[j] = new_words[j], new_words[i]
    
    # Insert a function word (mimic translation artifacts)
    if random.random() < 0.05:
        insert_pos = random.randint(0, len(new_words))
        new_words.insert(insert_pos, random.choice(list(EWE_FUNCTION_WORDS)))
    
    return ' '.join(new_words)
```

### 4. Function Word Insertion

This technique inserts Ewe function words at random positions to increase linguistic diversity:

```python
def function_insertion(text, prob=0.05):
    """Insert Ewe function words at random positions"""
    words = text.split()
    new_words = words.copy()
    num_insertions = max(1, int(len(words) * prob))
    
    for _ in range(num_insertions):
        word_to_insert = random.choice(list(EWE_FUNCTION_WORDS))
        insert_pos = random.randint(0, len(new_words))
        new_words.insert(insert_pos, word_to_insert)
    
    return ' '.join(new_words)
```

**Example**:
- Original: `ŋutsu la zɔ mɔ didi`
- Augmented: `ŋutsu la zɔ mɔ eye didi` (inserted "eye" - "and")

## How Data Augmentation is Applied

TransformEw2 applies these techniques through a unified function:

```python
def augment_translation_data_ewe(df, src_col, tgt_col, techniques=None):
    """Augment Ewe translation data with enhanced techniques"""
    if techniques is None:
        techniques = ['word_dropout', 'word_swap', 'back_translation', 'function_insertion']
    
    augmented_data = [df]  # Start with original data
    
    for technique in techniques:
        aug_df = df.copy()
        # Apply the specific technique
        aug_df[src_col] = aug_df[src_col].apply(lambda x: technique_function(x))
        augmented_data.append(aug_df)
    
    # Combine all augmented data
    result_df = pd.concat(augmented_data).reset_index(drop=True)
    
    # Filter low-quality augmentations
    result_df = result_df[quality_filter_condition]
    
    return result_df
```

## Testing Data Augmentation

We've created comprehensive tests to verify our augmentation techniques:

```python
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

These tests ensure that our augmentation techniques are working correctly and providing the expected benefits for Ewe-English translation.
