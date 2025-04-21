#!/usr/bin/env python3
"""
Training script for TransformEw2 model.
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.dataset_loader import create_dataloaders
from transformers import AutoTokenizer
from model import make_model, create_masks


class NoamOpt:
    """Optimizer with learning rate schedule from the paper"""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and learning rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement learning rate schedule"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) * \
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    """Implement label smoothing to prevent overconfidence"""

    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
        return loss.item() * norm


def train_epoch(model, data_loader, criterion, optimizer, scheduler, device, pad_idx=0, accum_iter=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    tokens = 0
    n_accum = 0
    start = time.time()

    for i, batch in enumerate(tqdm(data_loader, desc="Training")):
        src = batch["source"].to(device)
        tgt = batch["target"].to(device)
        tgt_y = tgt[:, 1:]
        tgt = tgt[:, :-1]

        # Create masks
        src_mask, tgt_mask = create_masks(src, tgt, pad_idx)

        # Forward pass
        out = model(src, tgt, src_mask, tgt_mask)

        # Calculate loss
        loss = criterion(out.contiguous().view(-1, out.size(-1)),
                         tgt_y.contiguous().view(-1))

        # Scale loss for gradient accumulation
        loss = loss / accum_iter

        # Backward pass
        loss.backward()

        # Count tokens for reporting
        ntokens = (tgt_y != pad_idx).data.sum().item()
        total_tokens += ntokens
        tokens += ntokens

        # Update weights if we've accumulated enough gradients
        if (i + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            n_accum += 1

        # Track loss
        total_loss += loss.item() * accum_iter

        # Report progress
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"Epoch Step: {i+1} | Loss: {total_loss/total_tokens:.4f} | "
                  f"Tokens per Sec: {tokens/elapsed:.1f}")
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def evaluate(model, data_loader, criterion, device, pad_idx=0):
    """Evaluate the model on validation data"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["source"].to(device)
            tgt = batch["target"].to(device)
            tgt_y = tgt[:, 1:]
            tgt = tgt[:, :-1]

            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt, pad_idx)

            # Forward pass
            out = model(src, tgt, src_mask, tgt_mask)

            # Calculate loss
            loss = criterion(out.contiguous().view(-1, out.size(-1)),
                             tgt_y.contiguous().view(-1))

            # Count tokens
            ntokens = (tgt_y != pad_idx).data.sum().item()
            total_tokens += ntokens
            total_loss += loss.item()

    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Train TransformEw2 model")
    parser.add_argument("--config", default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--data_dir", help="Data directory (overrides config)")
    parser.add_argument("--output_dir", help="Output directory (overrides config)")
    parser.add_argument("--no_stoplist", action="store_true", help="Disable stoplist filtering")
    parser.add_argument("--resume", help="Resume training from checkpoint")
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

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizers
    try:
        src_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(config['data']['data_dir'], f"{config['data']['src_lang']}_tokenizer")
        )
        tgt_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(config['data']['data_dir'], f"{config['data']['tgt_lang']}_tokenizer")
        )
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        print("Trying to load SentencePiece tokenizers instead...")
        from data_processing.load_tokenizers import load_sentencepiece_tokenizer

        src_sp_path = os.path.join(config['data']['data_dir'], f"{config['data']['src_lang']}_sp.model")
        tgt_sp_path = os.path.join(config['data']['data_dir'], f"{config['data']['tgt_lang']}_sp.model")

        src_tokenizer = load_sentencepiece_tokenizer(src_sp_path)
        tgt_tokenizer = load_sentencepiece_tokenizer(tgt_sp_path)

    # Get vocabulary sizes
    if hasattr(src_tokenizer, 'vocab_size'):
        src_vocab_size = src_tokenizer.vocab_size
    elif hasattr(src_tokenizer, 'get_piece_size'):
        src_vocab_size = src_tokenizer.get_piece_size()
    else:
        src_vocab_size = len(src_tokenizer)

    if hasattr(tgt_tokenizer, 'vocab_size'):
        tgt_vocab_size = tgt_tokenizer.vocab_size
    elif hasattr(tgt_tokenizer, 'get_piece_size'):
        tgt_vocab_size = tgt_tokenizer.get_piece_size()
    else:
        tgt_vocab_size = len(tgt_tokenizer)

    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")

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

    # Initialize model
    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        N=config['model']['encoder_layers'],
        d_model=config['model']['embedding_dim'],
        d_ff=config['model']['feedforward_dim'],
        h=config['model']['attention_heads'],
        dropout=config['model']['dropout']
    )
    model = model.to(device)

    # Define loss function with label smoothing
    criterion = LabelSmoothing(size=tgt_vocab_size, padding_idx=0, smoothing=0.1)
    criterion = criterion.to(device)

    # Define optimizer with learning rate schedule
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOpt(
        model_size=config['model']['embedding_dim'],
        factor=1.0,
        warmup=config['training']['warmup_steps'],
        optimizer=optimizer
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch, config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")

        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            pad_idx=0,
            accum_iter=config['training']['gradient_accumulation_steps']
        )
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate on validation set
        val_loss = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            pad_idx=0
        )
        print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(
            config['training']['output_dir'],
            f"transformew2_{config['data']['src_lang']}_{config['data']['tgt_lang']}_epoch{epoch+1}.pt"
        )
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'config': config
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = os.path.join(
                config['training']['output_dir'],
                f"transformew2_{config['data']['src_lang']}_{config['data']['tgt_lang']}_best.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'src_vocab_size': src_vocab_size,
                'tgt_vocab_size': tgt_vocab_size,
                'config': config
            }, best_model_path)
            print(f"Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")

        # Early stopping
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    final_model_path = os.path.join(
        config['training']['output_dir'],
        f"transformew2_{config['data']['src_lang']}_{config['data']['tgt_lang']}_final.pt"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
        'config': config
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

    # Evaluate on test set
    test_loss = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        pad_idx=0
    )
    print(f"\nTest Loss: {test_loss:.4f}")
    print("Training complete!")


if __name__ == "__main__":
    main()