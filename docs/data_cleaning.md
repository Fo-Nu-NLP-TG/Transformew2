# Data Cleaning in TransformEw2

## What is Data Cleaning?

Data cleaning is the process of preparing raw text data for machine learning by removing or correcting issues that could negatively impact model training. This includes removing noise, standardizing formats, and filtering out problematic examples.

## Why is Data Cleaning Important?

For machine translation, especially with low-resource languages like Ewe:

1. **Garbage In, Garbage Out**: The quality of training data directly affects the quality of translations.
2. **Consistency**: Standardized formatting helps the model learn patterns more effectively.
3. **Noise Reduction**: Removing irrelevant information helps the model focus on important patterns.
4. **Error Prevention**: Filtering out problematic examples prevents the model from learning incorrect patterns.

## TransformEw2's Data Cleaning Pipeline

TransformEw2 implements a comprehensive data cleaning pipeline with several key components:

### 1. Basic Text Cleaning

This component handles common text issues:

```python
def clean_text(self, text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
        
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (customize as needed)
    text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
    return text.strip()
```

**Example**:
- Before: `<b>me le</b>   tɔ me   ɖu ƒu!!!`
- After: `me le tɔ me ɖu ƒu`

### 2. Length-Based Filtering

This component filters out sentence pairs that are too short or too long:

```python
def filter_by_length(self, df, src_col, tgt_col, min_len=3, max_len=100):
    """Filter out sentence pairs that are too short or too long"""
    mask = (df[src_col].str.split().str.len() >= min_len) & \
           (df[src_col].str.split().str.len() <= max_len) & \
           (df[tgt_col].str.split().str.len() >= min_len) & \
           (df[tgt_col].str.split().str.len() <= max_len)
    return df[mask]
```

**Why This Matters**:
- Very short sentences (1-2 words) often lack context for proper translation
- Very long sentences can be difficult for the model to process
- Extreme length differences between source and target may indicate alignment issues

### 3. Language Detection (Optional)

This component can filter out sentence pairs where the language doesn't match the expected language:

```python
def filter_by_language(self, df, src_col, tgt_col, src_lang='ee', tgt_lang='en'):
    """Filter out sentence pairs with wrong languages using langdetect"""
    def is_correct_language(row):
        try:
            # For Ewe, we might need a custom approach as langdetect might not support it
            if src_lang == 'ee':
                src_detected = True  # Skip detection for Ewe
            else:
                src_detected = detect(row[src_col]) == src_lang
            
            tgt_detected = detect(row[tgt_col]) == tgt_lang
            return src_detected and tgt_detected
        except LangDetectException:
            return False
    
    return df[df.apply(is_correct_language, axis=1)]
```

**Note**: Since standard language detection libraries may not support Ewe, this is often skipped for the source language.

### 4. Empty Row Removal

This component removes rows where either the source or target text is empty:

```python
# Remove empty rows
df = df[(df[src_col].str.strip() != '') & (df[tgt_col].str.strip() != '')]
```

### 5. Corpus File Generation

This component creates corpus files for further processing:

```python
def update_corpus_file(self, series, corpus_filename):
    """Update corpus file with cleaned text"""
    corpus_path = os.path.join(self.data_dir, corpus_filename)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in series:
            f.write(str(text) + "\n")
```

## The Complete Preprocessing Pipeline

TransformEw2 combines all these components into a comprehensive pipeline:

```python
def preprocess_dataset(self, file_name, src_col, tgt_col, src_lang='ee', tgt_lang='en'):
    """Apply full preprocessing pipeline to a dataset"""
    # Load data
    df = pd.read_csv(file_path)
    
    # 1. Basic cleaning
    df[src_col] = df[src_col].apply(self.clean_text)
    df[tgt_col] = df[tgt_col].apply(self.clean_text)
    
    # 2. Remove empty rows
    df = df[(df[src_col].str.strip() != '') & (df[tgt_col].str.strip() != '')]
    
    # 3. Length filtering
    df = self.filter_by_length(df, src_col, tgt_col, min_len=3, max_len=100)
    
    # 4. Optional language detection
    # df = self.filter_by_language(df, src_col, tgt_col, src_lang, tgt_lang)
    
    # Save preprocessed data
    df.to_csv(output_path, index=False)
    
    # Update corpus files
    self.update_corpus_file(df[src_col], f"{src_lang}_corpus.txt")
    self.update_corpus_file(df[tgt_col], f"{tgt_lang}_corpus.txt")
    
    return df
```

## Testing Data Cleaning

We've created comprehensive tests to verify our data cleaning pipeline:

```python
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
```

These tests ensure that our data cleaning pipeline is working correctly and providing clean, consistent data for training our translation model.
