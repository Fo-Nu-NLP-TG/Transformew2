# TransformEw2: Overview

## What is TransformEw2?

TransformEw2 is an improved transformer-based model for Ewe-English translation. It builds on our previous implementation (TransformEw1) with significant enhancements to address the challenges of low-resource machine translation.

## Key Components

TransformEw2 consists of several key components:

1. **Stoplist Generation**: A system for identifying and filtering out common function words (like "the", "and", "of" in English) that don't carry much meaning but appear frequently.

2. **Data Augmentation**: Techniques to artificially expand our limited Ewe-English parallel data by creating variations of existing sentences.

3. **Data Cleaning**: Methods to clean and normalize text data, ensuring it's in a consistent format for training.

4. **Tokenization**: Converting text into tokens (subwords) that the model can process.

5. **Transformer Model**: The neural network architecture that learns to translate between languages.

## Key Improvements Over TransformEw1

TransformEw2 addresses several limitations of our previous implementation:

- **Better Stopword Handling**: More sophisticated identification of function words in both Ewe and English
- **Enhanced Data Augmentation**: Smarter techniques that preserve the meaning and structure of Ewe sentences
- **Improved Text Normalization**: Better handling of Ewe-specific characters and diacritics
- **Fixed Architecture Issues**: Corrected dimension handling and other technical problems
- **More Comprehensive Testing**: Rigorous testing of all components

## Project Structure

The project is organized into several directories:

- `data_processing/`: Contains modules for data preparation, cleaning, and augmentation
- `scripts/`: Utility scripts for preprocessing and training
- `configs/`: Configuration files for the model
- `tests/`: Test suites for verifying functionality
- `docs/`: Documentation (you are here!)
- `Transformew1/`: The previous implementation (for reference)

## Getting Started

See the main README.md file for installation and usage instructions.
