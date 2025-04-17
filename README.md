# TransformEw2: Transformer-based Ewe-English Translation

An improved transformer-based model for Ewe-English translation, building on lessons learned from our previous implementation.

## Key Improvements

- **Enhanced Data Processing**: Language-specific stopword handling and advanced text normalization
- **Optimized Tokenization**: Better subword segmentation for morphologically rich Ewe language
- **Fixed Architecture**: Corrected output dimension handling for proper vocabulary mapping
- **Training Stability**: Implemented gradient accumulation and improved initialization
- **Modular Design**: Clear separation of components with robust import structure

## Installation

```bash
# Clone the repository
git clone https://github.com/username/TransformEw2.git
cd TransformEw2

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
TransformEw2/
├── data_processing/           # Data preprocessing utilities
│   ├── create_splits.py       # Dataset splitting utilities
│   ├── data_augmentation.py   # Data augmentation techniques
│   ├── data_cleaner.py        # Text cleaning and filtering
│   ├── dataset_loader.py      # Dataset loading utilities
│   ├── dataset_splitter.py    # Dataset splitting implementation
│   ├── generate_stoplists.py  # Stoplist generation script
│   ├── load_tokenizers.py     # Tokenizer loading utilities
│   ├── run_tokenizer_training.py # Tokenizer training script
│   ├── stoplist_generator.py  # Stopword generation for Ewe/English
│   ├── tokenizer_trainer.py   # Tokenizer training utilities
│   └── transformer_dataset.py # Dataset preparation for transformer
├── scripts/                   # Utility scripts
│   ├── preprocess_data.py     # Data preprocessing pipeline
│   └── train_model.py         # Model training script
├── configs/                   # Configuration files
│   └── default.yaml           # Default configuration
├── tests/                     # Test suite
│   ├── __init__.py            # Makes tests a package
│   └── test_data_augmentation.py # Tests for data augmentation
├── Transformew1/              # Previous implementation (reference)
│   ├── encode_decode.py       # Encoding/decoding utilities
│   ├── inference.py           # Inference utilities
│   ├── model_utils.py         # Model utility functions
│   ├── training.py            # Training utilities
│   ├── train_transformer_cli.py # CLI for training
│   ├── train_transformer.py   # Training implementation
│   ├── translate.py           # Translation utilities
│   └── visualization.py       # Visualization utilities
├── docs/                      # Documentation
├── requirements.txt           # Project dependencies
├── LICENSE                    # MIT License
├── TRANSFORMEW1_ISSUES.md     # Issues with previous implementation
└── TRANSFORMEW2_IMPROVEMENTS.md # Detailed improvements documentation
```

## Usage

### Data Preprocessing

```bash
# Generate stoplists for Ewe and English
python scripts/preprocess_data.py --generate_stoplists

# Train tokenizers
python scripts/preprocess_data.py --train_tokenizers

# Process and prepare datasets
python scripts/preprocess_data.py --prepare_datasets
```

### Training

```bash
# Train with default configuration
python scripts/train_model.py

# Train with custom configuration
python scripts/train_model.py --config configs/custom.yaml
```

## Data Processing Pipeline

TransformEw2 implements a comprehensive data processing pipeline:

1. **Text Cleaning**: Removes noise and normalizes text
2. **Stopword Handling**: Applies language-specific stopword filtering
3. **Tokenization**: Uses optimized subword tokenization for both languages
4. **Augmentation**: Applies techniques like word dropout and swapping
5. **Dataset Creation**: Prepares batched datasets with proper padding and masking

## Data Augmentation

The project includes specialized data augmentation techniques for the Ewe language:

1. **Smart Word Dropout**: Selectively drops words while preserving verbs
2. **Smart Word Swap**: Swaps adjacent words without disrupting verb positions
3. **Simulated Back-Translation**: Mimics back-translation effects for low-resource languages
4. **Function Word Insertion**: Inserts Ewe function words to increase linguistic diversity

## Model Architecture

The transformer architecture follows the standard encoder-decoder design with:

- 6 encoder and 6 decoder layers
- 8 attention heads
- 512 embedding dimensions
- 2048 feed-forward dimensions
- Proper output dimension handling for vocabulary mapping

## Testing

```bash
# Run all tests
cd tests
python -m unittest discover

# Run specific test file
python -m unittest test_data_augmentation.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the FoNu NLP Technology Group for supporting this research
- Special thanks to contributors of Ewe language resources
