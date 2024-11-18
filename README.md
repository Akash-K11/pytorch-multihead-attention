# PyTorch Multi-Head Self-Attention

A clean, efficient implementation of the Multi-Head Self-Attention mechanism using PyTorch. This implementation includes visualization tools and is designed to be both educational and production-ready.

## Features

- Efficient implementation of Multi-Head Self-Attention
- Built-in attention weight visualization
- Support for attention masking
- Comprehensive testing and validation utilities

## Installation

```bash
git clone https://github.com/Akash-K11/pytorch-multihead-attention.git
cd pytorch-multihead-attention
pip install -r requirements.txt
```

## Implementation Details

### Architecture
The implementation follows the architecture described in "Attention Is All You Need" (Vaswani et al., 2017):

1. **Multi-Head Attention**
   - Allows the model to attend to different parts of the sequence simultaneously
   - Splits the input into multiple heads, each focusing on different aspects

2. **Scaled Dot-Product Attention**
   - Computes attention scores using scaled dot product
   - Applies softmax to get attention weights
   - Includes optional masking support

3. **Linear Projections**
   - Separate linear transformations for Query (Q), Key (K), and Value (V)
   - Final output projection to combine heads

### Key Components

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        # d_model: dimension of the model
        # num_heads: number of attention heads
```

## Visualization

The repository includes utilities for visualizing attention weights:

Example visualization output:
```
Query Position │
               │    ████████░░░░░░
               │    ░░████████░░░░
               │    ░░░░████████░░
               │    ░░░░░░████████
               └────────────────────
                    Key Position
```

## Performance Considerations

- Efficient batch processing using PyTorch operations
- Memory-efficient implementation of attention mechanism
- Optimized matrix operations

## Testing

Key test cases include:
- Attention weight computation
- Shape validation
- Masking functionality
- Gradient flow

## Acknowledgments

- Implementation inspired by the original Transformer paper