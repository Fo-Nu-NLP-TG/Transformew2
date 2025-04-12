# TransformEw2: Key Improvements

## Overview
TransformEw2 builds on our previous Ewe-English transformer implementation with significant enhancements to address the challenges of low-resource machine translation.

## Data Processing Improvements

### Enhanced Stoplist Generation
- **Dual-method approach**: Combined regex-based and frequency-based stopword detection
  - **Why**: Single detection methods miss many stopwords; combining approaches provides more comprehensive coverage
- **Language-specific patterns**: Custom regex patterns for Ewe function words, pronouns, and common verbs
  - **Why**: Generic patterns fail to capture Ewe-specific function words and grammatical markers
- **Threshold-based filtering**: Automatic identification of high-frequency, low-information words
  - **Why**: Frequency thresholds adapt to corpus characteristics, identifying stopwords that regex patterns might miss

## Usage
Refer to the main README.md for installation and usage instructions.

## Evaluation
Initial tests show significant improvements in BLEU scores and human evaluation metrics compared to TransformEw1.
