

This implementation of Rotary Positional Embeddings (RoPE) extends the original concept introduced by [Su et al.](https://arxiv.org/abs/2104.09864) with several adaptive mechanisms, including pitch-conditioning, variable radius, and continuous position interpolation through betweenness scoring.

The module can adapt the base frequency parameter (`theta`) according to pitch information, creating a perceptual mapping between fundamental frequency and positional encoding rate. This allows the model to dynamically adjust its attention mechanism based on audio characteristics.

Optionally enables learnable amplitudes for the rotations rather than fixed unit circles.
Variable radii are added in place of unit circle radius(1.0) associated with torch.polar. The frequencies (f0) are time aligned with tokens creating acoustically-weighted positional encodings where the "loudness" of each position in the embedding space reflects the acoustic prominence in the original speech.

    
