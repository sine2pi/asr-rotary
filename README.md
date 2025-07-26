

This implementation of Rotary Positional Embeddings (RoPE) extends the original concept introduced by [Su et al.](https://arxiv.org/abs/2104.09864) with several adaptive mechanisms, including pitch-conditioning, variable radius, and continuous position interpolation through betweenness scoring.

The module can adapt the base frequency parameter (`theta`) according to pitch information, creating a perceptual mapping between fundamental frequency and positional encoding rate. This allows the model to dynamically adjust its attention mechanism based on audio characteristics.

Optionally enables learnable amplitudes for the rotations rather than fixed unit circles.
Variable radii are added in place of unit circle radius(1.0) associated with torch.polar. The frequencies (f0) are time aligned with tokens creating acoustically-weighted positional encodings where the "loudness" of each position in the embedding space reflects the acoustic prominence in the original speech.

### Decrease WER by 20% compared to standard inverse frequency.

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000 

### Standared inv freqs:
     freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / (self.head_dim // 2)))

    
