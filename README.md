Extract frequencies from raw audio and map 1:1 to rotary frequency during training. Experiment

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
        self._last_f0 = None  # Store the last F0 value to detect changes
        
        # Fallback theta for when no F0 is provided
        self.base_theta = nn.Parameter(torch.tensor(float(theta)), requires_grad=False)
        self.register_buffer('inv_freq', 1.0 / (theta ** (torch.arange(0, dims, 2) / dims)))
        
        self.f0_scale = nn.Parameter(torch.tensor(25.0), requires_grad=learned_freq)
        
        # Min and max thetas to ensure values stay in reasonable range
        self.min_theta = 500.0  # Minimum theta value
        self.max_theta = 20000.0  # Maximum theta value
        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
        
    def forward(self, x=None, f0=None) -> Tensor:
        if f0 is not None:
            # Clean up the F0 representation and ensure it's usable
            f0_clean = torch.clamp(f0.squeeze(0) if f0.ndim == 3 else f0, min=1e-5)
            if f0_clean.ndim > 1:
                f0_clean = f0_clean.squeeze()
            
            # DIRECTLY use F0 as theta - no scaling or transformation
            # Each position directly gets its own F0 value
            scale = F.softplus(self.f0_scale)
            if isinstance(x, int):
                # If x is just a length, create position indices
                t_pos = torch.arange(x, device=f0_clean.device).float()
                
                # Make sure f0_clean matches the length of positions or adjust
                if f0_clean.shape[0] != x:
                    f0_clean = F.interpolate(f0_clean.unsqueeze(0), size=x, mode='linear').squeeze(0)
                
                # Each position gets its own set of frequencies based on its F0
                frequencies = []
                for pos in range(x):
                    # Get F0 for this position (frame)
                    pos_f0 = f0_clean[pos].item()
                    
                    # Scale F0 to appropriate theta range
                    pos_theta = torch.clamp(pos_f0 * scale, self.min_theta, self.max_theta)
                    
                    # Generate position-specific frequencies
                    pos_inv_freq = 1.0 / (pos_theta ** (torch.arange(0, self.dims, 2, 
                                        device=f0_clean.device) / self.dims))
                    
                    # Calculate frequencies for this position
                    freq = pos * pos_inv_freq + self.bias[pos][:pos_inv_freq.shape[0]]
                    frequencies.append(freq)
                
                # Stack to create the full frequency tensor
                pos_freqs = torch.stack(frequencies)
                
            else:
                # x is already a tensor of positions
                t_pos = x.float().to(f0_clean.device)
                
                # Make sure f0_clean matches the length of positions
                if f0_clean.shape[0] != t_pos.shape[0]:
                    f0_clean = F.interpolate(f0_clean.unsqueeze(0), size=t_pos.shape[0], mode='linear').squeeze(0)
                
                # Create position-specific frequencies with position-specific F0 values
                frequencies = []
                for pos in range(t_pos.shape[0]):
                    pos_f0 = f0_clean[pos].item()
                    pos_inv_freq = 1.0 / (pos_f0 ** (torch.arange(0, self.dims, 2, 
                                        device=f0_clean.device) / self.dims))
                    freq = t_pos[pos] * pos_inv_freq + self.bias[pos][:pos_inv_freq.shape[0]]
                    frequencies.append(freq)
                
                pos_freqs = torch.stack(frequencies)
        else:
            # Fallback to standard RoPE if no F0 provided
            inv_freq = self.inv_freq.to(self.base_theta.device)
            
            if isinstance(x, int):
                t_pos = torch.arange(x, device=inv_freq.device).float()
            else:
                t_pos = x.float().to(inv_freq.device)
            
            pos_freqs = torch.einsum('i,j->ij', t_pos, inv_freq)
            pos_freqs = pos_freqs + self.bias[:pos_freqs.shape[0]]
        
        # Convert to complex representation
        freqs = torch.polar(torch.ones_like(pos_freqs), pos_freqs)
        freqs = freqs.unsqueeze(0)
        
        # Debug printing - show the scaled F0 values
        if self.debug and f0 is not None:
            current_f0 = f0_clean[0].item()
            current_theta = current_f0 * scale.item()
            
            if self._last_f0 is None or abs(current_f0 - self._last_f0) > 1e-3:
                print(f"Raw F0: {current_f0:.2f} Hz → Scaled theta: {current_theta:.2f}")
                print(f"Using F0 directly as theta - sample F0: {current_f0:.2f} Hz")
                print(f"Frequency shape: {freqs.shape}")
                self._last_f0 = current_f0
            
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
