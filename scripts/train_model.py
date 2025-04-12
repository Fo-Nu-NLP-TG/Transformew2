#!/usr/bin/env python3
"""
Training script for TransformEw2 model.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.dataset_loader import create_dataloaders
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train TransformEw2 model")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--data_dir", help="Data directory (overrides config)")
    parser.add_argument("--output_dir", help="Output directory (overrides config)")
    parser.add_argument("--no_stoplist", action="store_true", help="Disable stoplist filtering")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command-line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    # Load tokenizers
    src_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config['data']['data_dir'], f"{config['data']['src_lang']}_tokenizer")
    )
    tgt_tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(config['data']['data_dir'], f"{config['data']['tgt_lang']}_tokenizer")
    )
    
    # Create dataloaders with stoplist filtering
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data']['data_dir'],
        src_lang=config['data']['src_lang'],
        tgt_lang=config['data']['tgt_lang'],
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length'],
        apply_stoplist=not args.no_stoplist  # Apply stoplist by default
    )
    
    print(f"Created dataloaders with {'stoplist filtering' if not args.no_stoplist else 'no stoplist filtering'}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model (to be implemented)
    print("Model initialization and training to be implemented")

if __name__ == "__main__":
    main()