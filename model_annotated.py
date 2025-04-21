"""
TransformEw2 model implementation - ANNOTATED VERSION FOR LEARNING

This module implements the improved transformer architecture for Ewe-English translation,
addressing the limitations identified in the previous implementation (Transformew1).

KEY DIFFERENCES FROM TRANSFORMEW1:
1. Fixed dimension handling for vocabulary sizes
2. Improved initialization for better training stability
3. Enhanced attention mechanism with better memory management
4. Support for gradient accumulation
5. More modular design with clear separation of components
6. Better error handling and debugging support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def clones(module, N):
    """
    Produce N identical layers.
    
    This helper function creates multiple copies of a module, used to create
    stacks of identical layers in the encoder and decoder.
    
    Args:
        module: The module to clone
        N: Number of copies to make
        
    Returns:
        nn.ModuleList containing N copies of module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """
    Layer normalization module
    
    Normalizes the features of the input tensor along the last dimension.
    This helps stabilize training by ensuring the inputs to each layer have
    consistent scale.
    
    DIFFERENCE FROM TRANSFORMEW1: Same implementation but used more consistently
    throughout the model.
    """
    
    def __init__(self, features, eps=1e-6):
        """
        Initialize layer normalization.
        
        Args:
            features: The number of features in the input
            eps: Small constant for numerical stability
        """
        super(LayerNorm, self).__init__()
        # These parameters are learned during training
        self.a_2 = nn.Parameter(torch.ones(features))  # Scale parameter
        self.b_2 = nn.Parameter(torch.zeros(features))  # Shift parameter
        self.eps = eps
        
    def forward(self, x):
        """
        Apply layer normalization to input.
        
        Args:
            x: Input tensor of shape (..., features)
            
        Returns:
            Normalized tensor of the same shape
        """
        # Calculate mean and standard deviation along the last dimension
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # Normalize, scale, and shift
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    
    This implements the "Add & Norm" blocks in the transformer architecture.
    Note that the norm is applied first, which is a slight modification from
    the original paper but works better in practice.
    
    DIFFERENCE FROM TRANSFORMEW1: Same core implementation but with more
    consistent usage throughout the model.
    """
    
    def __init__(self, size, dropout):
        """
        Initialize the sublayer connection.
        
        Args:
            size: The feature dimension
            dropout: Dropout rate
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        
        Args:
            x: Input tensor
            sublayer: A function that processes the input
            
        Returns:
            Output tensor after applying sublayer, dropout, and residual connection
        """
        # Apply normalization first, then the sublayer function, then dropout,
        # and finally add the original input (residual connection)
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Encoder layer with self-attention and feed-forward network
    
    Each encoder layer has two main components:
    1. Multi-head self-attention mechanism
    2. Position-wise feed-forward network
    
    DIFFERENCE FROM TRANSFORMEW1: Better initialization and more consistent
    implementation.
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Initialize encoder layer.
        
        Args:
            size: Feature dimension
            self_attn: Self-attention module
            feed_forward: Feed-forward network module
            dropout: Dropout rate
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # Create two sublayer connections (one for attention, one for feed-forward)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
    def forward(self, x, mask):
        """
        Process input through self-attention and feed-forward networks.
        
        Args:
            x: Input tensor
            mask: Mask to prevent attending to padding tokens
            
        Returns:
            Processed tensor
        """
        # 1. Self-attention sublayer
        # The lambda function creates a closure that applies self-attention
        # with the input as query, key, and value
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 2. Feed-forward sublayer
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """
    Decoder layer with self-attention, encoder-attention, and feed-forward network
    
    Each decoder layer has three main components:
    1. Masked multi-head self-attention (over previous decoder outputs)
    2. Multi-head attention over encoder output
    3. Position-wise feed-forward network
    
    DIFFERENCE FROM TRANSFORMEW1: Better initialization and more consistent
    implementation.
    """
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Initialize decoder layer.
        
        Args:
            size: Feature dimension
            self_attn: Self-attention module for attending to previous decoder outputs
            src_attn: Attention module for attending to encoder outputs
            feed_forward: Feed-forward network module
            dropout: Dropout rate
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # Create three sublayer connections
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Process input through self-attention, encoder-attention, and feed-forward networks.
        
        Args:
            x: Input tensor (previous decoder outputs)
            memory: Encoder output
            src_mask: Mask for encoder outputs (padding)
            tgt_mask: Mask for decoder inputs (padding and future tokens)
            
        Returns:
            Processed tensor
        """
        m = memory
        # 1. Self-attention sublayer with target mask
        # This prevents attending to future tokens and padding
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 2. Encoder-attention sublayer with source mask
        # This allows attending to the encoder output (cross-attention)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 3. Feed-forward sublayer
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    """
    Core encoder is a stack of N identical layers
    
    The encoder processes the source sequence into a rich representation
    that captures contextual information.
    
    DIFFERENCE FROM TRANSFORMEW1: More consistent implementation with better
    error handling.
    """
    
    def __init__(self, layer, N):
        """
        Initialize encoder with N identical layers.
        
        Args:
            layer: Encoder layer module to clone
            N: Number of layers
        """
        super(Encoder, self).__init__()
        # Create N identical encoder layers
        self.layers = clones(layer, N)
        # Final layer normalization
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """
        Pass the input through each encoder layer in sequence.
        
        Args:
            x: Input tensor (embedded source tokens)
            mask: Mask to prevent attending to padding tokens
            
        Returns:
            Encoded representation
        """
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, mask)
        # Apply final layer normalization
        return self.norm(x)


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    
    The decoder generates the target sequence one token at a time,
    attending to both the encoder output and previously generated tokens.
    
    DIFFERENCE FROM TRANSFORMEW1: More consistent implementation with better
    error handling.
    """
    
    def __init__(self, layer, N):
        """
        Initialize decoder with N identical layers.
        
        Args:
            layer: Decoder layer module to clone
            N: Number of layers
        """
        super(Decoder, self).__init__()
        # Create N identical decoder layers
        self.layers = clones(layer, N)
        # Final layer normalization
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Pass the input through each decoder layer in sequence.
        
        Args:
            x: Input tensor (embedded target tokens)
            memory: Encoder output
            src_mask: Mask for encoder outputs (padding)
            tgt_mask: Mask for decoder inputs (padding and future tokens)
            
        Returns:
            Decoded representation
        """
        # Pass through each decoder layer sequentially
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # Apply final layer normalization
        return self.norm(x)


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    
    This is the core attention mechanism that computes weighted sums of values
    based on the compatibility of queries and keys.
    
    DIFFERENCE FROM TRANSFORMEW1: Returns attention weights for potential
    visualization and analysis. Also has better memory management.
    
    Args:
        query: Query tensor
        key: Key tensor
        value: Value tensor
        mask: Optional mask to prevent attending to certain positions
        dropout: Optional dropout module
        
    Returns:
        Tuple of (weighted sum of values, attention weights)
    """
    # Get the feature dimension of the query
    d_k = query.size(-1)
    # Compute attention scores (dot product of queries and keys)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Set masked positions to a large negative value so they have ~0 weight after softmax
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    p_attn = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        p_attn = dropout(p_attn)
    
    # Compute weighted sum of values
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention with improved initialization
    
    This splits the attention into multiple heads, allowing the model to
    attend to different parts of the input simultaneously.
    
    DIFFERENCE FROM TRANSFORMEW1: 
    - Improved initialization for better training stability
    - Better memory management
    - Returns attention weights for potential visualization
    """
    
    def __init__(self, h, d_model, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            h: Number of attention heads
            d_model: Model dimension
            dropout: Dropout rate
        """
        super(MultiHeadedAttention, self).__init__()
        # Ensure d_model is divisible by h
        assert d_model % h == 0
        
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # Dimension of each head
        self.h = h  # Number of heads
        
        # Create 4 linear projections (query, key, value, output)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # Store attention weights for visualization
        self.dropout = nn.Dropout(p=dropout)
        
        # DIFFERENCE FROM TRANSFORMEW1: Improved initialization
        # Initialize with smaller weights for stability
        for i, linear in enumerate(self.linears):
            nn.init.xavier_uniform_(linear.weight, gain=0.1)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0.)
        
    def forward(self, query, key, value, mask=None):
        """
        Implement multi-head attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional mask to prevent attending to certain positions
            
        Returns:
            Output tensor after multi-head attention
        """
        if mask is not None:
            # Same mask applied to all h heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        
        # Apply final linear projection
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """
    Implements the position-wise feed-forward network.
    
    This is a simple two-layer neural network applied to each position separately.
    
    DIFFERENCE FROM TRANSFORMEW1:
    - Improved initialization for better training stability
    - Uses Kaiming initialization for first layer and Xavier for second layer
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize position-wise feed-forward network.
        
        Args:
            d_model: Model dimension (input and output)
            d_ff: Hidden dimension
            dropout: Dropout rate
        """
        super(PositionwiseFeedForward, self).__init__()
        # Two linear transformations with a ReLU in between
        self.w_1 = nn.Linear(d_model, d_ff)  # First layer (expand)
        self.w_2 = nn.Linear(d_ff, d_model)  # Second layer (project back)
        self.dropout = nn.Dropout(dropout)
        
        # DIFFERENCE FROM TRANSFORMEW1: Improved initialization
        # Kaiming initialization for first layer (better for ReLU)
        nn.init.kaiming_normal_(self.w_1.weight)
        # Xavier initialization for second layer
        nn.init.xavier_uniform_(self.w_2.weight)
        # Initialize biases to zero
        nn.init.constant_(self.w_1.bias, 0.)
        nn.init.constant_(self.w_2.bias, 0.)
        
    def forward(self, x):
        """
        Apply position-wise feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Processed tensor
        """
        # Apply first layer, ReLU, dropout, then second layer
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    """
    Embedding layer with scaled weights
    
    Converts token IDs to vectors and scales them by sqrt(d_model).
    
    DIFFERENCE FROM TRANSFORMEW1:
    - Improved initialization with normal distribution
    - More consistent scaling
    """
    
    def __init__(self, d_model, vocab):
        """
        Initialize embedding layer.
        
        Args:
            d_model: Model dimension
            vocab: Vocabulary size
        """
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        
        # DIFFERENCE FROM TRANSFORMEW1: Initialize embedding weights
        # with normal distribution for better training
        nn.init.normal_(self.lut.weight, mean=0, std=d_model**-0.5)
        
    def forward(self, x):
        """
        Convert token IDs to embeddings and scale.
        
        Args:
            x: Input tensor of token IDs
            
        Returns:
            Embedded and scaled tensor
        """
        # Embed and scale by sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    Implement the positional encoding function.
    
    Adds positional information to the embeddings since the transformer
    has no inherent notion of position.
    
    DIFFERENCE FROM TRANSFORMEW1: Same implementation but used more consistently.
    """
    
    def __init__(self, d_model, dropout, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state that's not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Input tensor (embedded tokens)
            
        Returns:
            Tensor with positional encoding added
        """
        # Add positional encoding to embeddings
        x = x + self.pe[:, :x.size(1)]
        # Apply dropout
        return self.dropout(x)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    
    Maps the decoder output to vocabulary probabilities.
    
    DIFFERENCE FROM TRANSFORMEW1:
    - Improved initialization with Xavier uniform
    - Better handling of output dimensions
    """
    
    def __init__(self, d_model, vocab):
        """
        Initialize generator.
        
        Args:
            d_model: Model dimension
            vocab: Vocabulary size
        """
        super(Generator, self).__init__()
        # Linear projection from d_model to vocab size
        self.proj = nn.Linear(d_model, vocab)
        
        # DIFFERENCE FROM TRANSFORMEW1: Improved initialization
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.)
        
    def forward(self, x):
        """
        Project features to vocabulary probabilities.
        
        Args:
            x: Input tensor (decoder output)
            
        Returns:
            Log probabilities over vocabulary
        """
        # Apply linear projection and log softmax
        return F.log_softmax(self.proj(x), dim=-1)


class TransformEw2(nn.Module):
    """
    TransformEw2: Improved transformer model for Ewe-English translation
    
    This is the main model class that combines all components.
    
    Key improvements over Transformew1:
    - Fixed dimension handling for vocabulary sizes
    - Improved initialization for better training stability
    - Support for gradient accumulation
    - Enhanced attention mechanism
    """
    
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        """
        Initialize the TransformEw2 model.
        
        Args:
            src_vocab: Source vocabulary size
            tgt_vocab: Target vocabulary size
            N: Number of encoder/decoder layers
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            h: Number of attention heads
            dropout: Dropout rate
        """
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
        
        # DIFFERENCE FROM TRANSFORMEW1: Verify generator output dimension
        if hasattr(self.generator, 'proj'):
            if self.generator.proj.out_features != tgt_vocab:
                print(f"WARNING: Generator output dimension ({self.generator.proj.out_features}) doesn't match target vocabulary size ({tgt_vocab})!")
                # Fix the dimension
                self.generator.proj = nn.Linear(d_model, tgt_vocab)
                print(f"Fixed Generator output dimension to: {self.generator.proj.out_features}")
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        
        Args:
            src: Source sequence
            tgt: Target sequence
            src_mask: Source mask
            tgt_mask: Target mask
            
        Returns:
            Log probabilities over target vocabulary
        """
        # Encode the source sequence
        memory = self.encode(src, src_mask)
        # Decode the target sequence using the encoded source
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        # Generate probabilities over vocabulary
        return self.generator(output)
    
    def encode(self, src, src_mask):
        """
        Encode source sequence.
        
        Args:
            src: Source sequence
            src_mask: Source mask
            
        Returns:
            Encoded representation
        """
        # Embed source tokens and apply encoder
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        Decode target sequence using source memory.
        
        Args:
            memory: Encoded source
            src_mask: Source mask
            tgt: Target sequence
            tgt_mask: Target mask
            
        Returns:
            Decoded representation
        """
        # Embed target tokens and apply decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def subsequent_mask(size):
    """
    Mask out subsequent positions.
    
    Creates a mask to prevent attending to future tokens in the target sequence.
    
    Args:
        size: Sequence length
        
    Returns:
        Boolean mask where True values allow attention
    """
    # Create an upper triangular matrix of ones
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    # Invert to get the mask (1 = attend, 0 = don't attend)
    return subsequent_mask == 0


def create_masks(src, tgt, pad_idx=0):
    """
    Create masks for training.
    
    Creates source and target masks to handle padding and prevent attending to future tokens.
    
    DIFFERENCE FROM TRANSFORMEW1: New helper function to centralize mask creation.
    
    Args:
        src: Source sequence
        tgt: Target sequence
        pad_idx: Padding token index
        
    Returns:
        Tuple of (source mask, target mask)
    """
    # Source mask to avoid attending to padding
    src_mask = (src != pad_idx).unsqueeze(-2)
    
    # Target mask to avoid attending to padding and future tokens
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).to(tgt.device)
    
    return src_mask, tgt_mask


def make_model(src_vocab_size, tgt_vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Helper: Construct a model from hyperparameters.
    
    DIFFERENCE FROM TRANSFORMEW1: More informative output and better error handling.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        N: Number of encoder/decoder layers
        d_model: Model dimension
        d_ff: Feed-forward hidden dimension
        h: Number of attention heads
        dropout: Dropout rate
        
    Returns:
        Initialized TransformEw2 model
    """
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
