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

        if axial and spec_shape is not None:
            time_frames, freq_bins = spec_shape
            self.time_frames = time_frames
            self.freq_bins = freq_bins
            
            time_theta = 50.0
            time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('time_freqs', time_freqs)
            
            freq_theta = 100.0
            freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('freq_freqs', freq_freqs)

    def pitch_bias(self, f0):
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
