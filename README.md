# Adaptive Rotary Positional Embeddings (RoPE)

## Overview

This implementation of Rotary Positional Embeddings (RoPE) extends the original concept introduced by [Su et al.](https://arxiv.org/abs/2104.09864) with several adaptive mechanisms, including pitch-conditioning, variable radius, and continuous position interpolation through betweenness scoring.

## Technical Details

### Core Functionality

The rotary module implements complex-valued rotational embeddings where each position is encoded as a rotation in the complex plane. This approach has several theoretical advantages:

- Relative position awareness without explicit pairwise calculations
- Decaying influence of distant tokens through rotation mathematics
- Enhanced extrapolation capabilities compared to absolute position encodings

### Key Extensions

#### Pitch-Adaptive Rotations

```python
f0_mean = f0_tensor.mean()
f0_mean = torch.clamp(f0_mean, min=80.0, max=600.0)
perceptual = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 600.0 / 700.0))
f0_theta = self.theta * (1.0 + perceptual)
```

The module can adapt the base frequency parameter (`theta`) according to pitch information, creating a perceptual mapping between fundamental frequency and positional encoding rate. This allows the model to dynamically adjust its attention mechanism based on audio characteristics.

#### Betweenness and Continuous Positioning

The module introduces a novel "betweenness" calculation that measures how much each token sits between others in the embedding space:

```python
direct = dist[fbatch, fsrc, ftgt].view(batch, s-2)
path1 = dist[fbatch, fsrc, fmid].view(batch, s-2)
path2 = dist[fbatch, fmid, ftgt].view(batch, s-2)
            
path = path1 + path2
scores = torch.relu(1.0 - (path - direct) / torch.clamp(direct, min=1e-6))
```

This betweenness score is used to create continuous interpolated positions, allowing tokens to effectively occupy non-integer positions in the sequence:

```python
a = self.btw_g * (btw - 0.5) * 2.0
pa = torch.clamp(pos + a, 0, self.max_ctx - 1)
```
Note that everything is experimental and in pre beta stage. This option works as it should but hasent been optimized. At the moment, a few of the vectors are a little large and will eat more vram than they should.

#### Variable Radius

Optionally enables learnable amplitudes for the rotations rather than fixed unit circles:

```python
if self.vradius:
    radius = F.softplus(self.radius)
    freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
```
Not sure on this one yet..

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `dims` | Dimension of the rotary embeddings (typically `head_dim`) |
| `max_ctx` | Maximum context length for precomputing frequencies |
| `theta` | Base frequency scaling factor (default: 10000) |
| `lfreq` | Whether inverse frequencies are learnable |
| `vradius` | Whether to use variable radius for embeddings |
| `lradius` | Whether radius values are learnable |
| `ltheta` | Whether theta parameter is learnable |
| `lpitch` | Whether pitch scaling is learnable |

## Usage

### Basic Usage

```python
# Initialize the rotary embeddings module
rope = rotary(dims=64, max_ctx=2048)

# Get embeddings for a sequence
freqs = rope(seq_len)

# Apply embeddings to query/key tensors
q_rotated = rope.apply_rotary(q, freqs)
k_rotated = rope.apply_rotary(k, freqs)
```

### Pitch-Conditioned Usage

```python
# With pitch information (fundamental frequency)
freqs = rope(seq_len, f0=pitch_tensor)

# Apply to attention components
q_rotated = rope.apply_rotary(q, freqs)
k_rotated = rope.apply_rotary(k, freqs)
```

### Betweenness-Adjusted Positioning

```python
# Calculate betweenness scores
btw = rope.get_btw(x)

# Apply rotary embeddings with continuous positioning
x_rotated = rope.btw_rope(x, btw)
```

## Implementation Notes

- Complex number operations utilize PyTorch's complex tensor support
- Pitch adaptation uses a log-based perceptual scaling to map frequency to rotation rate
- Betweenness calculations use a triangular distance comparison method
- Position interpolation uses a gather-based approach for efficient continuous sampling

## Debugging

The module includes built-in debug capabilities that can log theta values and tensor shapes during forward passes:

```python
# Enable rotary debugging
rope = rotary(dims=64, debug=["rotary"])
```

## Requirements

- PyTorch 1.8+ (for complex tensor support)
- CUDA-compatible device recommended for performance
