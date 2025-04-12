# FoNu Transformer V2

An improved transformer-based model for Ewe-English translation, building on lessons learned from our previous implementation.

## Key Improvements

- Fixed output dimension handling for proper vocabulary mapping
- Modular architecture with clear separation of components
- Robust import structure that works across environments
- Improved training stability with gradient accumulation
- Enhanced data processing pipeline for low-resource languages

## Installation

```bash
# Clone the repository
git clone https://github.com/FoNuNLPTG/FoNu_Transformer_V2.git
cd FoNu_Transformer_V2

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
FoNu_Transformer_V2/
├── fonu/                   # Main package
│   ├── __init__.py         # Package initialization
│   ├── model/              # Model architecture
│   │   ├── __init__.py
│   │   ├── transformer.py  # Core transformer implementation
│   │   ├── encoder.py      # Encoder implementation
│   │   ├── decoder.py      # Decoder implementation
│   │   └── attention.py    # Attention mechanisms
│   ├── data/               # Data processing
│   │   ├── __init__.py
│   │   ├── dataset.py      # Dataset classes
│   │   ├── tokenizer.py    # Tokenization utilities
│   │   └── augmentation.py # Data augmentation techniques
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── train_utils.py  # Training utilities
│       └── metrics.py      # Evaluation metrics
├── scripts/                # Training and inference scripts
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── translate.py        # Translation script
├── configs/                # Configuration files
│   ├── default.yaml        # Default configuration
│   └── experiments/        # Experiment-specific configs
├── tests/                  # Unit tests
├── examples/               # Example usage
├── requirements.txt        # Dependencies
└── setup.py                # Package setup
```

## Usage

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Translation

```bash
python scripts/translate.py --model path/to/model --input "Ewe text to translate"
```

## Documentation

See the [docs](docs/) directory for detailed documentation.

## License

MIT License