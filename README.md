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
│   ├── data_cleaner.py        # Text cleaning and filtering
│   ├── stoplist_generator.py  # Stopword generation for Ewe/English
│   ├── tokenizer_trainer.py   # Tokenizer training utilities
│   ├── data_augmentation.py   # Data augmentation techniques
│   └── dataset_loader.py      # Dataset loading and preparation
├── model/                     # Model architecture
│   ├── transformer.py         # Core transformer implementation
│   ├── encoder.py             # Encoder implementation
│   ├── decoder.py             # Decoder implementation
│   └── attention.py           # Attention mechanisms
├── training/                  # Training utilities
│   ├── trainer.py             # Training loop implementation
│   ├── optimizer.py           # Custom optimizers and schedules
│   └── metrics.py             # Evaluation metrics
├── scripts/                   # Utility scripts
│   ├── preprocess_data.py     # Data preprocessing pipeline
│   ├── train_model.py         # Model training script
│   └── translate.py           # Translation inference script
├── configs/                   # Configuration files
│   └── default.yaml           # Default configuration
├── requirements.txt           # Project dependencies
└── TRANSFORMEW2_IMPROVEMENTS.md  # Detailed improvements documentation
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

### Translation

```bash
# Translate a single sentence
python scripts/translate.py --input "Ewe text to translate"

# Translate a file
python scripts/translate.py --input_file path/to/input.txt --output_file path/to/output.txt
```

## Data Processing Pipeline

TransformEw2 implements a comprehensive data processing pipeline:

1. **Text Cleaning**: Removes noise and normalizes text
2. **Stopword Handling**: Applies language-specific stopword filtering
3. **Tokenization**: Uses optimized subword tokenization for both languages
4. **Augmentation**: Applies techniques like word dropout and swapping
5. **Dataset Creation**: Prepares batched datasets with proper padding and masking

## Model Architecture

The transformer architecture follows the standard encoder-decoder design with:

- 6 encoder and 6 decoder layers
- 8 attention heads
- 512 embedding dimensions
- 2048 feed-forward dimensions
- Proper output dimension handling for vocabulary mapping

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to the FoNu NLP Technology Group for supporting this research
- Special thanks to contributors of Ewe language resources
