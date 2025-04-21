#!/usr/bin/env python3
"""
Inference script for TransformEw2 model.
"""

import argparse
import os
import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model import TransformEw2, create_masks
from data_processing.load_tokenizers import load_sentencepiece_tokenizer
from transformers import AutoTokenizer


def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device):
    """Greedily decode a sequence"""
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    
    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = create_masks(src, ys)[1]
        
        # Forward pass
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        
        # Get next token
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # Add next token to output
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word).to(device)], dim=1)
        
        # Stop if end symbol is generated
        if next_word == end_symbol:
            break
    
    return ys


def translate(model, src_text, src_tokenizer, tgt_tokenizer, max_len=100, device="cpu"):
    """Translate a source text to target language"""
    model.eval()
    
    # Tokenize source text
    if hasattr(src_tokenizer, 'encode'):
        # Hugging Face tokenizer
        src_tokens = src_tokenizer.encode(src_text).ids
    else:
        # SentencePiece tokenizer
        src_tokens = src_tokenizer.encode(src_text, out_type=int)
        # Add BOS if not already added
        if len(src_tokens) > 0 and src_tokens[0] != 2:  # BOS id
            src_tokens = [2] + src_tokens
        # Add EOS if not already added
        if len(src_tokens) > 0 and src_tokens[-1] != 3:  # EOS id
            src_tokens = src_tokens + [3]
    
    # Convert to tensor
    src = torch.tensor([src_tokens]).to(device)
    
    # Create source mask
    src_mask = (src != 0).unsqueeze(-2)
    
    # Decode
    out = greedy_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        max_len=max_len,
        start_symbol=2,  # BOS token
        end_symbol=3,    # EOS token
        device=device
    )
    
    # Convert output tokens to text
    out_tokens = out[0].tolist()
    
    # Remove BOS and EOS tokens
    if 2 in out_tokens:
        out_tokens.remove(2)  # Remove BOS
    if 3 in out_tokens:
        out_tokens = out_tokens[:out_tokens.index(3)]  # Remove EOS and everything after
    
    # Decode tokens to text
    if hasattr(tgt_tokenizer, 'decode'):
        # Hugging Face tokenizer
        out_text = tgt_tokenizer.decode(out_tokens)
    else:
        # SentencePiece tokenizer
        out_text = tgt_tokenizer.decode(out_tokens)
    
    return out_text


def main():
    parser = argparse.ArgumentParser(description="Inference with TransformEw2 model")
    parser.add_argument("--model_path", required=True, help="Path to the trained model")
    parser.add_argument("--src_tokenizer", help="Path to the source tokenizer")
    parser.add_argument("--tgt_tokenizer", help="Path to the target tokenizer")
    parser.add_argument("--text", help="Text to translate")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum output length")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get model configuration
    if 'config' in checkpoint:
        config = checkpoint['config']
        src_vocab_size = checkpoint['src_vocab_size']
        tgt_vocab_size = checkpoint['tgt_vocab_size']
        
        # Create model
        model = TransformEw2(
            src_vocab=src_vocab_size,
            tgt_vocab=tgt_vocab_size,
            N=config['model']['encoder_layers'],
            d_model=config['model']['embedding_dim'],
            d_ff=config['model']['feedforward_dim'],
            h=config['model']['attention_heads'],
            dropout=config['model']['dropout']
        )
    else:
        # Legacy model format
        src_vocab_size = checkpoint['src_vocab_size']
        tgt_vocab_size = checkpoint['tgt_vocab_size']
        
        # Create model with default parameters
        model = TransformEw2(
            src_vocab=src_vocab_size,
            tgt_vocab=tgt_vocab_size
        )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load tokenizers
    if args.src_tokenizer and args.tgt_tokenizer:
        # Use provided tokenizer paths
        src_tokenizer_path = args.src_tokenizer
        tgt_tokenizer_path = args.tgt_tokenizer
    elif 'config' in checkpoint:
        # Use paths from config
        data_dir = config['data']['data_dir']
        src_lang = config['data']['src_lang']
        tgt_lang = config['data']['tgt_lang']
        
        # Try Hugging Face tokenizers first
        src_tokenizer_path = os.path.join(data_dir, f"{src_lang}_tokenizer")
        tgt_tokenizer_path = os.path.join(data_dir, f"{tgt_lang}_tokenizer")
        
        # If not found, try SentencePiece tokenizers
        if not os.path.exists(src_tokenizer_path):
            src_tokenizer_path = os.path.join(data_dir, f"{src_lang}_sp.model")
        if not os.path.exists(tgt_tokenizer_path):
            tgt_tokenizer_path = os.path.join(data_dir, f"{tgt_lang}_sp.model")
    else:
        print("Error: No tokenizer paths provided and no config found in checkpoint")
        return
    
    # Load tokenizers
    try:
        # Try loading as Hugging Face tokenizers
        src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_path)
        tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_tokenizer_path)
    except:
        # Fall back to SentencePiece tokenizers
        src_tokenizer = load_sentencepiece_tokenizer(src_tokenizer_path)
        tgt_tokenizer = load_sentencepiece_tokenizer(tgt_tokenizer_path)
    
    # Interactive mode
    if args.interactive:
        print("Interactive mode. Enter text to translate (Ctrl+C to exit):")
        try:
            while True:
                text = input("> ")
                if not text.strip():
                    continue
                
                translation = translate(
                    model=model,
                    src_text=text,
                    src_tokenizer=src_tokenizer,
                    tgt_tokenizer=tgt_tokenizer,
                    max_len=args.max_len,
                    device=device
                )
                print(f"Translation: {translation}")
        except KeyboardInterrupt:
            print("\nExiting...")
    
    # Single text mode
    elif args.text:
        translation = translate(
            model=model,
            src_text=args.text,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            max_len=args.max_len,
            device=device
        )
        print(f"Source: {args.text}")
        print(f"Translation: {translation}")
    
    # Example mode
    else:
        examples = [
            "me le tɔ me ɖu ƒu",
            "ɖevi la ɖu nu nyuie",
            "akpe na wò ɖe kpekpeɖeŋu la ta"
        ]
        
        print("Translating example sentences:")
        for example in examples:
            translation = translate(
                model=model,
                src_text=example,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                max_len=args.max_len,
                device=device
            )
            print(f"Source: {example}")
            print(f"Translation: {translation}")
            print()


if __name__ == "__main__":
    main()
