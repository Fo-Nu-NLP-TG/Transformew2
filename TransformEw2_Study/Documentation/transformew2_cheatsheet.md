# TransformEw2 Cheat Sheet

## Core Components

| Component | Purpose | Improvements in Transformew2 |
|-----------|---------|------------------------------|
| `TransformEw2` | Main model class | Fixed dimension handling, improved initialization |
| `Encoder` | Processes source sequence | More consistent implementation |
| `Decoder` | Generates target sequence | More consistent implementation |
| `MultiHeadedAttention` | Attention mechanism | Xavier initialization (gain=0.1), returns attention weights |
| `PositionwiseFeedForward` | Feed-forward network | Kaiming init for first layer, Xavier for second |
| `Embeddings` | Token to vector conversion | Normal distribution initialization |
| `PositionalEncoding` | Adds position information | Same as Transformew1 |
| `Generator` | Maps to vocabulary | Xavier initialization, dimension verification |

## Key Functions

| Function | Purpose | Improvements in Transformew2 |
|----------|---------|------------------------------|
| `make_model()` | Creates model instance | Better error handling, informative output |
| `attention()` | Core attention calculation | Returns attention weights |
| `create_masks()` | Creates attention masks | New helper function |
| `subsequent_mask()` | Masks future tokens | Same as Transformew1 |
| `clones()` | Creates multiple copies | Same as Transformew1 |

## Initialization Techniques

| Component | Transformew1 | Transformew2 |
|-----------|--------------|--------------|
| Attention projections | Xavier uniform | Xavier uniform (gain=0.1) |
| Feed-forward (layer 1) | Xavier uniform | Kaiming normal |
| Feed-forward (layer 2) | Xavier uniform | Xavier uniform |
| Embeddings | Default | Normal (mean=0, std=d_model^-0.5) |
| Generator | Default | Xavier uniform |
| Biases | Default | Constant (0) |

## Dimension Handling

```python
# Transformew2 verification
if hasattr(self.generator, 'proj'):
    if self.generator.proj.out_features != tgt_vocab:
        print(f"WARNING: Generator output dimension doesn't match target vocabulary size!")
        # Fix the dimension
        self.generator.proj = nn.Linear(d_model, tgt_vocab)
```

## Forward Pass Flow

1. **Encode**: `src → src_embed → encoder → memory`
2. **Decode**: `tgt → tgt_embed → decoder(memory) → output`
3. **Generate**: `output → generator → log_probabilities`

## Attention Mechanism

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

**Multi-Head Attention**:
1. Project Q, K, V to h different spaces
2. Apply attention in each space
3. Concatenate results
4. Apply final projection

## Training Improvements

| Feature | Transformew1 | Transformew2 |
|---------|--------------|--------------|
| Learning rate | Basic scheduler | Noam scheduler with warmup |
| Label smoothing | Basic | Enhanced with better padding handling |
| Gradient accumulation | No | Yes |
| Early stopping | No | Yes |
| Checkpointing | Basic | Comprehensive (best model, regular) |

## Inference Improvements

| Feature | Transformew1 | Transformew2 |
|---------|--------------|--------------|
| Greedy decoding | Basic | Enhanced |
| Beam search | No | Yes |
| Interactive mode | Basic | Enhanced |

## Key Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| N | 6 | Number of encoder/decoder layers |
| d_model | 512 | Model dimension |
| d_ff | 2048 | Feed-forward hidden dimension |
| h | 8 | Number of attention heads |
| dropout | 0.1 | Dropout rate |

## Common Issues and Solutions

| Issue | Transformew1 Solution | Transformew2 Solution |
|-------|----------------------|----------------------|
| Dimension mismatch | Manual debugging | Automatic verification and fixing |
| Training instability | Basic initialization | Component-specific initialization |
| Memory limitations | Reduce batch size | Gradient accumulation |
| Overfitting | Basic regularization | Early stopping, better regularization |

## Code Structure

```
model.py
├── Helper Functions
│   ├── clones()
│   ├── attention()
│   ├── subsequent_mask()
│   └── create_masks()
├── Core Components
│   ├── LayerNorm
│   ├── SublayerConnection
│   ├── EncoderLayer
│   ├── DecoderLayer
│   ├── Encoder
│   ├── Decoder
│   ├── MultiHeadedAttention
│   ├── PositionwiseFeedForward
│   ├── Embeddings
│   ├── PositionalEncoding
│   └── Generator
├── Main Model
│   └── TransformEw2
└── Model Creation
    └── make_model()
```

## Quick Reference for Differences

1. **Fixed Dimension Handling**: Automatic verification and correction
2. **Improved Initialization**: Component-specific initialization techniques
3. **Enhanced Attention**: Returns attention weights, better memory management
4. **Gradient Accumulation**: Support for effectively larger batch sizes
5. **Modular Design**: Better separation of components, helper functions
6. **Error Handling**: More informative output, better debugging support
