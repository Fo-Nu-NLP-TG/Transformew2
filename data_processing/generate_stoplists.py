#!/usr/bin/env python3
"""
Script to generate stoplists for Ewe and English languages.
"""

from stoplist_generator import StoplistGenerator
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate stoplists for Ewe and English")
    parser.add_argument("--data_dir", default="./data/processed", help="Directory with corpus files")
    parser.add_argument("--output_dir", default="./data/processed", help="Directory to save stoplists")
    args = parser.parse_args()
    
    # Check if corpus files exist
    ewe_corpus_path = os.path.join(args.data_dir, "ewe_corpus.txt")
    english_corpus_path = os.path.join(args.data_dir, "english_corpus.txt")
    
    if not os.path.exists(ewe_corpus_path) or not os.path.exists(english_corpus_path):
        print("Corpus files not found. Please run data_cleaner.py first.")
        return
    
    # Load corpus files
    with open(ewe_corpus_path, 'r', encoding='utf-8') as f:
        ewe_corpus = f.read()
    
    with open(english_corpus_path, 'r', encoding='utf-8') as f:
        english_corpus = f.read()
    
    # Generate stoplists
    print("Generating stoplists...")
    generator = StoplistGenerator()
    ewe_stopwords, english_stopwords = generator.generate_stoplists(ewe_corpus, english_corpus)
    
    # Save stoplists
    generator.save_stoplists(args.output_dir)
    
    print(f"Generated {len(ewe_stopwords)} Ewe stopwords and {len(english_stopwords)} English stopwords")
    print(f"Stoplists saved to {args.output_dir}")

if __name__ == "__main__":
    main()