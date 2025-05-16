
Extract frequencies from raw audio and map 1:1 to rotary frequency during training. Experiment

Intuition:

Different pitch regions should have different positional encoding characteristics
Higher pitch → faster rotations → positions become more distinct
Lower pitch → slower rotations → positions blend more smoothly

Linguistic: Pitch patterns in speech carry meaningful prosodic information (questions, emphasis, emotion). By making positional encodings F0-aware, we help the model distinguish these patterns.

Position-Specific Processing: Using different thetas for different positions based on their F0. 
This allows the model to:

Process high-pitched regions differently from low-pitched regions
Pay attention to pitch transitions
Potentially capture speech prosody better than fixed positional encodings

Replace the standard fixed theta with F0-based theta:

In high-pitched regions: faster rotations → positions become more distinct
In low-pitched regions: slower rotations → positions blend more smoothly
This effectively creates a "pitch-adaptive attention field" where the model's attention patterns dynamically adjust based on pitch characteristics.

``` python


    # --- Parosody ---
    if parosody:
        hop_length = extractor.hop_length
        win_length = 256
        wav = wav.unsqueeze(0) if wav.ndim == 1 else wav  
        device = wav.device if wav.is_cuda else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wav = wav.to(device)
        f0 = torchcrepe.predict(wav, sampling_rate, model="tiny")
        # f0 = torchyin.estimate(wav, sample_rate=sampling_rate, frame_stride=hop_length / sampling_rate)
        f0 = torch.nan_to_num(f0, nan=0.0)
        # f0 = (f0 - f0.mean()) / (f0.std() + 1e-8)
        num_frames = f0.shape[-1]

        energies = []
        for i in range(num_frames):
            start = int(i * hop_length)
            end = int(start + win_length)
            frame = wav[0, start:end]
            if frame.numel() == 0:
                rms = 0.0
                power = 0.0
            else:
                rms = torch.sqrt(torch.mean(frame ** 2) + 1e-8).item()
                hann = torch.hann_window(frame.numel(), device=frame.device)
                windowed = frame * hann
                power = torch.sum(windowed ** 2).item()
            energies.append([rms, power])
        energy = torch.tensor(energies, dtype=torch.float32, device=f0.device).T  # shape: (2, num_frames)

        max_len = max(f0.shape[-1], energy.shape[-1])
        if f0.shape[-1] < max_len:
            f0 = F.pad(f0, (0, max_len - f0.shape[-1]), value=0.0)
        if energy.shape[-1] < max_len:
            energy = F.pad(energy, (0, max_len - energy.shape[-1]), value=0.0)

        blend = torch.sigmoid(torch.tensor(0.5, device=f0.device))
        parosody = blend * f0 + (1 - blend) * energy
        target_len = current_features.shape[-1]
        parosody = match_length(parosody, target_len)
        batch["parosody"] = parosody#.cpu()

        if f0_contour:
            f0 = match_length(f0, target_len)
            batch["f0_contour"] = f0
        if energy_contour:
            energy = match_length(energy, target_len)
            batch["energy_contour"] = energy

class F0RotaryDirect(nn.Module):
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, debug=False):
        super().__init__()
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self._last_f0 = None
        
        # Base parameters
        self.base_theta = nn.Parameter(torch.tensor(float(theta)), requires_grad=False)
        self.register_buffer('inv_freq', 1.0 / (theta ** (torch.arange(0, dims, 2) / dims)))
        
        # F0 scaling parameter
        self.f0_scale = nn.Parameter(torch.tensor(25.0), requires_grad=learned_freq)
        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
        
    def forward(self, x=None, f0=None) -> Tensor:
        # Handle input format
        if isinstance(x, int):
            t_pos = torch.arange(x, device=self.inv_freq.device).float()
        else:
            t_pos = x.float().to(self.inv_freq.device)
        
        if f0 is not None:
            # Simplify F0 handling - just extract a single value
            f0_clean = f0.squeeze(0) if f0.ndim == 3 else f0
            if f0_clean.ndim > 1:
                f0_clean = f0_clean.squeeze()
                
            # Use mean F0 as the theta modifier
            mean_f0 = f0_clean.mean().clamp(min=20.0)
            scale = F.softplus(self.f0_scale)
            
            # Calculate modified theta
            f0_theta = mean_f0 * scale
            
            # Clamp within reasonable range
            f0_theta = torch.clamp(f0_theta, min=500.0, max=20000.0)
            
            # Efficiently calculate inverse frequencies
            inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, 
                              device=f0_clean.device) / self.dims))
        else:
            # Use standard RoPE frequencies when no F0 is present
            inv_freq = self.inv_freq.to(self.base_theta.device)
        
        # Efficiently calculate frequencies using einsum
        pos_freqs = torch.einsum('i,j->ij', t_pos, inv_freq)
        pos_freqs = pos_freqs + self.bias[:pos_freqs.shape[0]]
        
        # Convert to complex representation
        freqs = torch.polar(torch.ones_like(pos_freqs), pos_freqs)
        freqs = freqs.unsqueeze(0)
        
        # Optional debug info
        if self.debug and f0 is not None and self._counter < 5:
            print(f"Using mean F0: {mean_f0:.2f} Hz → Scaled theta: {f0_theta:.2f}")
            self._counter += 1
            
        return freqs
    
    @staticmethod
    def apply_rotary(x, freqs):
        multihead_format = len(freqs.shape) == 4
        if multihead_format:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)
        else:
            x1 = x[..., :freqs.shape[-1]*2]
            x2 = x[..., freqs.shape[-1]*2:]
            x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
            x1 = torch.view_as_complex(x1)
            x1 = x1 * freqs
            x1 = torch.view_as_real(x1).flatten(-2)
            return torch.cat([x1.type_as(x), x2], dim=-1)

```
