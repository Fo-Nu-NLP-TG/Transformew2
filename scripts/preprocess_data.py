#!/usr/bin/env python3
"""
Comprehensive data preprocessing pipeline for TransformEw2.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.data_cleaner import DataCleaner
from data_processing.stoplist_generator import StoplistGenerator
from data_processing.tokenizer_trainer import TokenizerTrainer
from data_processing.data_augmentation import augment_translation_data

def main():
    parser = argparse.ArgumentParser(description="TransformEw2 Data Preprocessing Pipeline")
    parser.add_argument("--data_dir", default="./data/processed", help="Directory with data files")
    parser.add_argument("--output_dir", default="./data/processed", help="Directory to save processed data")
    parser.add_argument("--generate_stoplists", action="store_true", help="Generate stoplists for Ewe and English")
    parser.add_argument("--clean_data", action="store_true", help="Clean and preprocess raw data")
    parser.add_argument("--train_tokenizers", action="store_true", help="Train tokenizers")
    parser.add_argument("--augment_data", action="store_true", help="Apply data augmentation")
    parser.add_argument("--prepare_datasets", action="store_true", help="Prepare final datasets for training")
    parser.add_argument("--run_all", action="store_true", help="Run the complete pipeline")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run all steps if --run_all is specified
    if args.run_all:
        args.clean_data = True
        args.generate_stoplists = True
        args.train_tokenizers = True
        args.augment_data = True
        args.prepare_datasets = True
    
    # Step 1: Clean and preprocess raw data
    if args.clean_data:
        print("\n=== Step 1: Cleaning and preprocessing raw data ===")
        cleaner = DataCleaner(data_dir=args.data_dir)
        cleaner.preprocess_all_datasets()
    
    # Step 2: Generate stoplists
    if args.generate_stoplists:
        print("\n=== Step 2: Generating stoplists ===")
        # Check if corpus files exist
        ewe_corpus_path = os.path.join(args.data_dir, "ewe_corpus.txt")
        english_corpus_path = os.path.join(args.data_dir, "english_corpus.txt")
        
        if not os.path.exists(ewe_corpus_path) or not os.path.exists(english_corpus_path):
            print("Corpus files not found. Please run with --clean_data first.")
            if not args.clean_data:
                return
        
        # Load corpus files
        with open(ewe_corpus_path, 'r', encoding='utf-8') as f:
            ewe_corpus = f.read()
        
        with open(english_corpus_path, 'r', encoding='utf-8') as f:
            english_corpus = f.read()
        
        # Generate stoplists
        print("Generating stoplists using regex and frequency methods...")
        generator = StoplistGenerator()
        ewe_stopwords, english_stopwords = generator.generate_stoplists(ewe_corpus, english_corpus)
        
        # Save stoplists
        generator.save_stoplists(args.output_dir)
        
        print(f"Generated {len(ewe_stopwords)} Ewe stopwords and {len(english_stopwords)} English stopwords")
        print(f"Stoplists saved to {args.output_dir}")
        
        # Apply stoplist filtering to corpus files
        print("Applying stoplist filtering to corpus files...")
        apply_stoplist_filtering(args.data_dir, ewe_stopwords, english_stopwords)
    
    # Step 3: Train tokenizers
    if args.train_tokenizers:
        print("\n=== Step 3: Training tokenizers ===")
        tokenizer_trainer = TokenizerTrainer(data_dir=args.data_dir)
        
        # Check which language corpora are available
        languages = []
        for lang in ['ewe', 'english', 'french']:
            corpus_path = os.path.join(args.data_dir, f"{lang}_corpus.txt")
            if os.path.exists(corpus_path):
                languages.append(lang)
                print(f"Found corpus for {lang}")
        
        if not languages:
            print("No corpus files found. Please run with --clean_data first.")
            if not args.clean_data:
                return
        
        # Train BPE tokenizers
        for lang in languages:
            print(f"\nTraining BPE tokenizer for {lang}...")
            tokenizer_trainer.train_huggingface_tokenizer(
                lang, 
                model_type='bpe',
                vocab_size=16000 if lang == 'ewe' else 32000
            )
    
    # Step 4: Apply data augmentation
    if args.augment_data:
        print("\n=== Step 4: Applying data augmentation ===")
        # This step would use the data_augmentation module to apply techniques
        # like word dropout and word swapping
        print("Data augmentation functionality to be implemented")
    
    # Step 5: Prepare final datasets
    if args.prepare_datasets:
        print("\n=== Step 5: Preparing final datasets ===")
        # This step would prepare the final datasets for training
        print("Dataset preparation functionality to be implemented")
    
    print("\nPreprocessing pipeline completed!")

def apply_stoplist_filtering(data_dir, ewe_stopwords, english_stopwords):
    """Apply stoplist filtering to corpus files"""
    # Filter Ewe corpus
    ewe_corpus_path = os.path.join(data_dir, "ewe_corpus.txt")
    filtered_ewe_path = os.path.join(data_dir, "ewe_corpus_filtered.txt")
    
    with open(ewe_corpus_path, 'r', encoding='utf-8') as f:
        ewe_lines = f.readlines()
    
    with open(filtered_ewe_path, 'w', encoding='utf-8') as f:
        for line in ewe_lines:
            words = line.split()
            filtered_words = [word for word in words if word.lower() not in ewe_stopwords]
            f.write(' '.join(filtered_words) + '\n')
    
    # Filter English corpus
    english_corpus_path = os.path.join(data_dir, "english_corpus.txt")
    filtered_english_path = os.path.join(data_dir, "english_corpus_filtered.txt")
    
    with open(english_corpus_path, 'r', encoding='utf-8') as f:
        english_lines = f.readlines()
    
    with open(filtered_english_path, 'w', encoding='utf-8') as f:
        for line in english_lines:
            words = line.split()
            filtered_words = [word for word in words if word.lower() not in english_stopwords]
            f.write(' '.join(filtered_words) + '\n')
    
    print(f"Filtered corpus files saved to {filtered_ewe_path} and {filtered_english_path}")

if __name__ == "__main__":
    main()