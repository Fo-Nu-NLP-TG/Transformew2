# TransformEw2 Documentation

Welcome to the TransformEw2 documentation! This guide will help you understand the key components and improvements in our Ewe-English translation system.

## Contents

1. [Overview](transformew2_overview.md) - Introduction to TransformEw2 and its key components
2. [Stoplist Generation](stoplist_generation.md) - How we identify and handle function words
3. [Data Augmentation](data_augmentation.md) - Techniques for artificially expanding training data
4. [Data Cleaning](data_cleaning.md) - Methods for preparing and normalizing text data
5. [Testing Approach](testing_approach.md) - How we verify the functionality of TransformEw2

## Quick Start

If you're new to TransformEw2, we recommend starting with the [Overview](transformew2_overview.md) to get a high-level understanding of the system.

## Key Improvements

TransformEw2 includes several key improvements over the previous version:

1. **Enhanced Stoplist Generation**
   - Dual-method approach combining regex and frequency-based detection
   - Language-specific patterns for Ewe function words
   - Threshold-based filtering for high-frequency, low-information words

2. **Improved Data Augmentation**
   - Smart word dropout that preserves verbs
   - Smart word swap that maintains grammatical structure
   - Simulated back-translation for low-resource scenarios
   - Function word insertion for increased linguistic diversity

3. **Better Data Cleaning**
   - Improved HTML tag removal
   - Better whitespace normalization
   - Enhanced special character handling
   - More sophisticated length-based filtering

## Diagrams

### Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                      Raw Text Data                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Cleaning                          │
│   - HTML tag removal                                        │
│   - Whitespace normalization                                │
│   - Special character handling                              │
│   - Length filtering                                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Stoplist Generation                      │
│   - Regex-based detection                                   │
│   - Frequency-based detection                               │
│   - Combined approach                                       │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Data Augmentation                       │
│   - Smart word dropout                                      │
│   - Smart word swap                                         │
│   - Simulated back-translation                              │
│   - Function word insertion                                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Processed Dataset                        │
│   Ready for model training                                  │
└─────────────────────────────────────────────────────────────┘
```

### Stoplist Generation Approach

```
┌───────────────────────────────────────────────────────────────┐
│                  Dual-Method Stoplist Generation              │
└───────────────────────────┬───────────────────────────────────┘
                            │
        ┌──────────────────┴───────────────────┐
        ▼                                      ▼
┌─────────────────┐                  ┌──────────────────┐
│  Regex-based    │                  │  Frequency-based │
│    Method       │                  │     Method       │
└────────┬────────┘                  └────────┬─────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────────┐        ┌──────────────────────┐
│ Language-specific       │        │ Statistical analysis │
│ patterns for function   │        │ of word frequencies  │
│ words, pronouns, etc.   │        │ with thresholds      │
└─────────────────────────┘        └──────────────────────┘
         │                                    │
         └────────────────┬──────────────────┘
                          ▼
              ┌─────────────────────┐
              │  Combined Stoplist  │
              └─────────────────────┘
```

## Further Reading

- [TRANSFORMEW1_ISSUES.md](../TRANSFORMEW1_ISSUES.md) - Issues identified in the previous version
- [TRANSFORMEW2_IMPROVEMENTS.md](../TRANSFORMEW2_IMPROVEMENTS.md) - Detailed improvements in TransformEw2
- [README.md](../README.md) - Project overview and usage instructions
