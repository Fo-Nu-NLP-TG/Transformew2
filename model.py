"""
TransformEw2 model implementation.

This module implements the improved transformer architecture for Ewe-English translation,
addressing the limitations identified in the previous implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# The selected code defines a helper function clones that creates multiple identical 
# copies of a neural network module
def clones(module, N):
    # Takes a Pytorch module and an integer N as input
    # Creates N deep copies of the module using copy.deepcopy()
    # Returns these copies as a Pytorch nn.ModuleList
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# The selected code defines a class LayerNorm that implements the layer normalization module.
class LayerNorm(nn.Module):
    """Layer normalization module"""
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder layer with self-attention and feed-forward network"""
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """Decoder layer with self-attention, encoder-attention, and feed-forward network"""
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""
    
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Generic N layer decoder with masking."""
    
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# Scaled Dot-Product Attention
def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    # Gets the dimension of the query's last axis (the feature dimension), 
    # which is used for scaling.
    d_k = query.size(-1)
    # Computes attention scores by:
    # 1. Transposing the key matrix (swapping last two dimensions)
    # 2. Multiplying query with transposed key (dot product)
    # 3. Scaling by dividing by square root of dimension to prevent extremely small gradients
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Applies masking by setting scores to a very negative value (-1e9) 
    # where mask is 0, effectively making those positions have near-zero 
    # attention weight after softmax.
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # Applies softmax to convert scores into probabilities (attention weights) 
    # that sum to 1 along the last dimension.
    p_attn = F.softmax(scores, dim=-1)
    # Applies dropout to attention weights for regularization, 
    # randomly zeroing some weights to prevent overfitting.
    if dropout is not None:
        p_attn = dropout(p_attn)
    # Returns:
    # 1. The weighted sum of values (multiplying attention weights with value matrix)
    # 2. The attention weights themselves (useful for visualization or further processing)
    return torch.matmul(p_attn, value), p_attn

# Multi-Head Attention
class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention with improved initialization"""
    
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # Dimension of each head
        self.h = h # Number of heads
        # Calculate dimension per head
        # Create 4 linear projections (query, key, value, output)
        # Store attention weights for visualization
        # Dropout for regularization
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
        # Create 4 linear projections (query, key, value, outpu
        # Improved initialization for attention projections
        for i, linear in enumerate(self.linears):
            # Initialize with smaller weights for stability
            nn.init.xavier_uniform_(linear.weight, gain=0.1)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0.)
        
    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Improved initialization
        nn.init.kaiming_normal_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.constant_(self.w_1.bias, 0.)
        nn.init.constant_(self.w_2.bias, 0.)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """Embedding layer with scaled weights"""
    
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
        # Initialize embedding weights with normal distribution
        nn.init.normal_(self.lut.weight, mean=0, std=d_model**-0.5)
        
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""
    
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        
        # Initialize with Xavier uniform for better gradient flow
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.)
        
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class TransformEw2(nn.Module):
    """
    TransformEw2: Improved transformer model for Ewe-English translation
    
    Key improvements:
    - Fixed dimension handling for vocabulary sizes
    - Improved initialization for better training stability
    - Support for gradient accumulation
    - Enhanced attention mechanism
    """
    
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(TransformEw2, self).__init__()
        c = copy.deepcopy
        
        # Create attention, feed-forward, and positional encoding modules
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        
        # Create encoder and decoder
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        
        # Create embeddings for source and target
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        
        # Create generator
        self.generator = Generator(d_model, tgt_vocab)
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Store vocabulary sizes for verification
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Verify generator output dimension
        if hasattr(self.generator, 'proj'):
            if self.generator.proj.out_features != tgt_vocab:
                print(f"WARNING: Generator output dimension ({self.generator.proj.out_features}) doesn't match target vocabulary size ({tgt_vocab})!")
                # Fix the dimension
                self.generator.proj = nn.Linear(d_model, tgt_vocab)
                print(f"Fixed Generator output dimension to: {self.generator.proj.out_features}")
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        return self.generator(output)
    
    def encode(self, src, src_mask):
        """Encode source sequence."""
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        """Decode target sequence using source memory."""
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def create_masks(src, tgt, pad_idx=0):
    """Create masks for training."""
    # Source mask to avoid attending to padding
    src_mask = (src != pad_idx).unsqueeze(-2)
    
    # Target mask to avoid attending to padding and future tokens
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).to(tgt.device)
    
    return src_mask, tgt_mask


def make_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    model = TransformEw2(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        N=N,
        d_model=d_model,
        d_ff=d_ff,
        h=h,
        dropout=dropout
    )
    
    # Print model information
    print(f"Created TransformEw2 model with:")
    print(f"  - Source vocabulary size: {src_vocab_size}")
    print(f"  - Target vocabulary size: {tgt_vocab_size}")
    print(f"  - Encoder/Decoder layers: {N}")
    print(f"  - Model dimension: {d_model}")
    print(f"  - Feed-forward dimension: {d_ff}")
    print(f"  - Attention heads: {h}")
    print(f"  - Dropout: {dropout}")
    
    return model

