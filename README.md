

This implementation of Rotary Positional Embeddings (RoPE) extends the original concept introduced by [Su et al.](https://arxiv.org/abs/2104.09864) with several adaptive mechanisms, including pitch-conditioning, variable radius, and continuous position interpolation through betweenness scoring.

The module can adapt the base frequency parameter (`theta`) according to pitch information, creating a perceptual mapping between fundamental frequency and positional encoding rate. This allows the model to dynamically adjust its attention mechanism based on audio characteristics.

Optionally enables learnable amplitudes for the rotations rather than fixed unit circles.
Variable radii are added in place of unit circle radius(1.0) associated with torch.polar. The frequencies (f0) are time aligned with tokens creating acoustically-weighted positional encodings where the "loudness" of each position in the embedding space reflects the acoustic prominence in the original speech.

``` python
### Simple implimentation 

class rotary(nn.Module):
    """
    Experimental rotary embedding that modulates rotation radius based on f0 (pitch) contour.
    """
    def __init__(n, dims, head):
        super().__init__()
        n.dims = dims
        n.head = head
        n.head_dim = dims // head
        n.theta = nn.Parameter(torch.tensor(10000.0), requires_grad=True)
        n.lna = nn.LayerNorm(dims)
        n.register_buffer('freqs_base', n._compute_freqs_base(), persistent=False)

    def _compute_freqs_base(n):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 4000/200)), n.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 200 * mel_scale / 1000

    def forward(n, x, xa = None):
        b, h, c, d = x.shape

        t = torch.arange(c, device=device, dtype=dtype)
        freqs = torch.outer(t, n.freqs_base.to(device, dtype))
        freqs = freqs.view(1, 1, c, n.head_dim // 2)

        # if xa is not None:
        #     freqs = (torch.arange(0, x.shape[2], device=device))[:, None] * (xa * n.theta / 220.0) * n.freqs_base        
        #     freqs = (freqs + torch.pi) % (2 * torch.pi) - torch.pi 

        if xa is not None:
            radius = 1.0 + xa[:, :, :n.head_dim // 2]
            freqs = torch.polar(radius, freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)

        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)

## longer version

class rotary(nn.Module):
    def __init__(self, dims, head, max_ctx=1500, radii=False, debug: List[str] = [], use_pbias=False, axial=False, spec_shape=None):

        super(rotary, self).__init__()
        self.use_pbias = use_pbias
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.radii = radii
        self.debug = debug
        self.counter = 0
        self.last_theta = None
        self.axial = axial

        self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2), requires_grad=True if use_pbias else False)
        theta = (torch.tensor(10000, device=device, dtype=dtype))
        self.theta = nn.Parameter(theta, requires_grad=True)    
        self.theta_values = []

        if axial and spec_shape is not None: # for 2d spectrograms
            time_frames, freq_bins = spec_shape
            self.time_frames = time_frames
            self.freq_bins = freq_bins
            
            time_theta = 50.0
            time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('time_freqs', time_freqs)
            
            freq_theta = 100.0
            freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('freq_freqs', freq_freqs)

    def pitch_bias(self, f0): # meh
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)))
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def _apply_radii(self, freqs, f0, ctx):
        if self.radii and f0 is not None:
            radius = f0.to(device, dtype)
            # this simple method might not be worse than interpolation
            # L = radius.shape[0]
            # if L != ctx:
            #     F = L / ctx
            #     idx = torch.arange(ctx, device=f0.device)
            #     idx = (idx * F).long().clamp(0, L - 1)
            #     radius = radius[idx]
            #     return torch.polar(radius.unsqueeze(-1), freqs), radius
            # else:
            return torch.polar(radius.unsqueeze(-1), freqs), radius
        else:
            return torch.polar(torch.ones_like(freqs), freqs), None

    def check_f0(self, f0, f0t, ctx):
        if f0 is not None and f0.shape[1] == ctx:
            return f0
        elif f0t is not None and f0t.shape[1] == ctx:
            return f0t
        else:
            return None         

    def axial_freqs(self, ctx):
        if not self.axial:
            return None
        time_frames = self.time_frames
        freq_bins = self.freq_bins
    
        t = torch.arange(ctx, device=device, dtype=dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, self.time_freqs)
        freqs_y = torch.outer(t_y, self.freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    def _compute_freqs_base(self):
        mel_scale = torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 2000/80)), self.head_dim // 2, device=device, dtype=dtype) / 2595) - 1
        return 80 * mel_scale / 1000  

    def forward(self, x, ctx, en=None, rope: List[str] = ["4"]) -> Tensor:
     
        if "1" in rope: # Standard RoPE 
            freqs = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=dtype) / (self.head_dim // 2)))
        if "2" in rope: # 200Hz - 4000Hz (covers 95% of speech content)
            freqs = (self.theta / 220.0) * 200 * self.mel_scale_200_4000 / 1000
        if "3" in rope: # 150Hz - 6000Hz (covers speech + some emotion/intonation)
            freqs = (self.theta / 220.0) * 150 * self.mel_scale_150_6000 / 1000
        if "4" in rope: # 80Hz - 2000Hz (focus on fundamental frequencies + first few harmonics)
            freqs = (self.theta / 220.0) * 80 * self.mel_scale_80_2000 / 1000

        f0 = en.get("f0") if en is not None else None 
        f0t = en.get("f0t") if en is not None else None 

        f0 = self.check_f0(f0, f0t, ctx)
        if f0 is not None:
            # if f0.dim() == 2:
            #     f0 = f0.squeeze(0) 
            theta = f0 + self.theta  
        else:
            theta = self.theta 
        freqs = self.theta_freqs(theta)
        t = torch.arange(ctx, device=device, dtype=dtype)
        freqs = t[:, None] * freqs
        freqs, radius = self._apply_radii(freqs, f0, ctx)

        if self.axial and f == "spectrogram":
            freqs_2d = self.axial_freqs(ctx)
            if freqs_2d is not None:
                return freqs_2d.unsqueeze(0)

        if "radius" in self.debug and self.counter == 10:
            print(f"  [{layer}] [Radius] {radius.shape if radius is not None else None} {radius.mean() if radius is not None else None} [Theta] {theta.mean() if theta is not None else None} [f0] {f0.shape if f0 is not None else None} [Freqs] {freqs.shape} {freqs.mean():.2f} [ctx] {ctx}")
        self.counter += 1
        return freqs.unsqueeze(0)

    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        orig_shape = x1.shape
        if x1.ndim == 2:
            x1 = x1.unsqueeze(0)
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
        x1 = torch.view_as_complex(x1) * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        x1 = x1.view(orig_shape)
        return torch.cat([x1.type_as(x), x2], dim=-1)


## the two wild and crazy guys version

class Rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, learned_freq=True, 
                 use_freq_bands=False, speech_enhanced=False,
                 variable_radius=False, learned_radius=True, init_radius=1.0):
        super().__init__()
        self.dims = dims
        self.use_freq_bands = use_freq_bands
        self.variable_radius = variable_radius
        
        # Configure frequency parameters
        if not use_freq_bands:
            # Original implementation
            self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, dims, 2) / dims)),
                requires_grad=learned_freq
            )
            self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
            
            # Global radius parameter (if variable)
            if variable_radius:
                self.radius = nn.Parameter(
                    torch.ones(dims // 2) * init_radius,
                    requires_grad=learned_radius
                )
        else:
            # FrequencyBand implementation
            band_size = dims // 6  # Each band gets 1/3 of dims (x2 for complex numbers)
            
            # Low frequencies (0-500Hz range in speech)
            self.low_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Mid frequencies (500-2000Hz in speech)
            self.mid_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(band_size, 2*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # High frequencies (>2000Hz in speech)
            self.high_freq_audio = nn.Parameter(
                1.0 / (10000 ** (torch.arange(2*band_size, 3*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Text-specific high frequencies
            self.high_freq_text = nn.Parameter(
                1.0 / (10000 ** (torch.arange(2*band_size, 3*band_size, 2) / dims)),
                requires_grad=learned_freq
            )
            
            # Frequency-specific biases
            if speech_enhanced:
                self.low_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.mid_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
                self.high_bias = nn.Parameter(torch.zeros(max_ctx, band_size // 2))
            else:
                self.bias = nn.Parameter(torch.zeros(max_ctx, dims // 2))
            
            # Band-specific radius parameters (if variable)
            if variable_radius:
                self.low_radius = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.mid_radius = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.high_radius_audio = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                self.high_radius_text = nn.Parameter(
                    torch.ones(band_size // 2) * init_radius,
                    requires_grad=learned_radius
                )
                
        self.speech_enhanced = speech_enhanced and use_freq_bands

    def forward(self, positions, domain="audio", snr_estimate=None):
        if isinstance(positions, int):
            t = torch.arange(positions, device=self.get_device()).float()
        else:
            t = positions.float().to(self.get_device())
        
        if not self.use_freq_bands:
            # Original implementation
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            freqs = freqs + self.bias[:freqs.shape[0]]
            
            if self.variable_radius:
                # Apply learnable radius instead of fixed radius=1
                radius = F.softplus(self.radius)  # Ensure radius is positive
                freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
            else:
                # Original fixed radius
                freqs = torch.polar(torch.ones_like(freqs), freqs)
        else:
            # FrequencyBand implementation
            low = torch.einsum('i,j->ij', t, self.low_freq)
            mid = torch.einsum('i,j->ij', t, self.mid_freq)
            
            # Domain-specific high frequencies
            if domain == "audio":
                high = torch.einsum('i,j->ij', t, self.high_freq_audio)
            else:
                high = torch.einsum('i,j->ij', t, self.high_freq_text)
            
            # Apply bias
            if self.speech_enhanced:
                low = low + self.low_bias[:low.shape[0]]
                mid = mid + self.mid_bias[:mid.shape[0]]
                high = high + self.high_bias[:high.shape[0]]
            else:
                # Create full bias-adjusted frequencies before applying radius
                freqs = torch.cat([low, mid, high], dim=-1)
                freqs = freqs + self.bias[:freqs.shape[0]]
                low, mid, high = torch.split(freqs, freqs.shape[1]//3, dim=1)
            
            # Apply variable radius if enabled
            if self.variable_radius:
                # Get appropriate radius for each band
                low_radius = F.softplus(self.low_radius)
                mid_radius = F.softplus(self.mid_radius)
                
                if domain == "audio":
                    high_radius = F.softplus(self.high_radius_audio)
                else:
                    high_radius = F.softplus(self.high_radius_text)
                
                # Adjust radius based on SNR if provided (audio mode only)
                if snr_estimate is not None and domain == "audio":
                    # Convert SNR to a scaling factor (lower SNR = smaller high freq radius)
                    snr_factor = torch.sigmoid((snr_estimate - 5) / 5)  # Maps to 0-1
                    
                    # Apply progressively stronger scaling to higher frequencies
                    # (high frequencies most affected by noise)
                    low_radius = low_radius  # Low frequencies mostly preserved
                    mid_radius = mid_radius * (0.5 + 0.5 * snr_factor)  # Partial scaling
                    high_radius = high_radius * snr_factor  # Strongest scaling
                
                # Create complex numbers with variable radius for each band
                low_complex = torch.polar(low_radius.unsqueeze(0).expand_as(low), low)
                mid_complex = torch.polar(mid_radius.unsqueeze(0).expand_as(mid), mid)
                high_complex = torch.polar(high_radius.unsqueeze(0).expand_as(high), high)
                
                # Combine all bands
                freqs = torch.cat([low_complex, mid_complex, high_complex], dim=-1)
            else:
                # Use fixed radius=1 (original behavior)
                freqs = torch.cat([low, mid, high], dim=-1)
                freqs = torch.polar(torch.ones_like(freqs), freqs)
                
        return freqs
    
    def get_device(self):
        """Helper to get device from any parameter"""
        if hasattr(self, 'inv_freq'):
            return self.inv_freq.device
        return self.low_freq.device
        
    @staticmethod
    def apply_rotary(x, freqs):
        x1 = x[..., :freqs.shape[-1]*2]
        x2 = x[..., freqs.shape[-1]*2:]
        x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous() 
        x1 = torch.view_as_complex(x1)
        x1 = x1 * freqs
        x1 = torch.view_as_real(x1).flatten(-2)
        return torch.cat([x1.type_as(x), x2], dim=-1)

#####

def pitch_tokens(wav, t, labels, f0)
        wav = torch.from_numpy(wavnp)
        t2 = torch.from_numpy(t)
        audio_duration = len(wav) / sample_rate
        T = len(labels)
        tok_dur_sec = audio_duration / T
        token_starts = torch.arange(T) * tok_dur_sec
        token_ends = token_starts + tok_dur_sec
        start_idx = torch.searchsorted(t2, token_starts, side="left")
        end_idx = torch.searchsorted(t2, token_ends, side="right")
        pitch_tok = torch.zeros(T, dtype=torch.float32)
        for i in range(T):
            lo, hi = start_idx[i], max(start_idx[i]+1, end_idx[i]) # type: ignore
            segment = f0_np[lo:hi]
            if mode == "mean":
                pitch_tok[i] = segment.mean()
            elif mode == "median":
                pitch_tok[i] = torch.median(segment)
            else:
                pitch_tok[i] = segment[-1]
        pitch_tok[pitch_tok < 100.0] = 0.0
        bos_pitch = pitch_tok[0] if len(pitch_tok) > 0 else 0.0
        f0t_tensor = torch.cat([torch.tensor([bos_pitch]), pitch_tok])
        f0t = torch.where(f0t_tensor == 0.0, torch.zeros_like(f0t_tensor), (f0t_tensor - 71.0) / (500.0 - 71.0))
    return f0t


```
    
