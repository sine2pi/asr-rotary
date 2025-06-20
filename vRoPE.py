
class rotary(nn.Module):
    _seen = set()  
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, radii=False,
                 learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = [], 
                 use_pbias=False, use_2d_axial=False, spec_shape=None):
        super().__init__()

        self.use_pbias = False
        self.use_2d_axial = use_2d_axial
        self.spec_shape = spec_shape
        self.last_f0_theta = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32 
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.radii = radii
        f0_factor = 0.5
        self.learned_adaptation: bool = False
        pitch_scale = 1.0
        radius = 1
        
        if self.learned_adaptation:
            self.f0_scale = nn.Parameter(torch.tensor(f0_factor, device=self.device, dtype=self.dtype), requires_grad=True)
        else:
            self.register_buffer('f0_scale', torch.tensor(f0_factor))

        self.theta = nn.Parameter(torch.tensor(theta, device=self.device, dtype=self.dtype), requires_grad=True)
        self.pitch_scale = nn.Parameter(torch.tensor(pitch_scale, device=self.device, dtype=self.dtype), requires_grad=True)
        
        if use_2d_axial and spec_shape is not None:
            time_frames, freq_bins = spec_shape
            self.time_frames = time_frames
            self.freq_bins = freq_bins
            
            time_theta = 50.0
            time_freqs = 1.0 / (time_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('time_freqs', time_freqs)
            
            freq_theta = 100.0
            freq_freqs = 1.0 / (freq_theta ** (torch.arange(0, dims, 4)[:(dims // 4)].float() / dims))
            self.register_buffer('freq_freqs', freq_freqs)
        else:
            freqs = 1. / (theta ** (torch.arange(0, dims, 2, device=self.device, dtype=self.dtype)[:(dims // 2)].float() / dims))
            self.freqs = nn.Parameter(torch.tensor(freqs, device=self.device, dtype=self.dtype), requires_grad=True)
        self.radius = nn.Parameter(torch.ones(radius, device=self.device, dtype=self.dtype), requires_grad=True)

    def compute_2d_axial_freqs(self, seq_len):
        if not self.use_2d_axial:
            return None
        time_frames = self.time_frames
        freq_bins = self.freq_bins
    
        t = torch.arange(seq_len, device=self.device, dtype=self.dtype)
        t_x = (t % time_frames).float()
        t_y = torch.div(t, time_frames, rounding_mode='floor').float()
        freqs_x = torch.outer(t_x, self.time_freqs)
        freqs_y = torch.outer(t_y, self.freq_freqs)
        freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
        freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
        return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    def align_f0(self, f0, ctx):
        b, l = f0.shape
        if l == ctx:
            return f0.squeeze(0).float()  
        frames_per_token = l / ctx
        idx = torch.arange(ctx, device=self.device, dtype=self.dtype)
        src_idx = (idx * frames_per_token).long().clamp(0, l-1)
        batch_idx = torch.arange(b, device=self.device, dtype=self.dtype).unsqueeze(1)
        f0 = f0[batch_idx, src_idx]
        return f0.squeeze(0).float()

    def get_pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)) * self.pitch_scale)
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def forward(self, x=None, f0=None, layer=None, input_type="audio") -> Tensor:
        if isinstance(x, int):
            ctx = x
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            batch, ctx, dims = x.shape
        else:
            batch, head, ctx, head_dim = x.shape
            
        if self.use_2d_axial and input_type == "spectrogram":
            freqs_2d = self.compute_2d_axial_freqs(ctx)
            if freqs_2d is not None:
                return freqs_2d.unsqueeze(0)
                
        t = torch.arange(ctx, device=self.device, dtype=self.dtype)

        if f0 is not None:
            f0_mean = f0.mean() + 1e-8
            theta = f0_mean * self.pitch_scale
            freqs = 1.0 / (theta ** (torch.arange(0, self.dims, 2, device=self.device, dtype=self.dtype)[:(self.dims // 2)].float() / self.dims))
            if "rotary" in self.debug:
                print(f"{layer}: {theta:.2f} : {f0_mean:.2f} : {ctx} ")
        else:
            freqs = self.freqs
            
        freqs = t[:, None] * freqs[None, :]
        
        if self.radii:
            if f0 is not None:
                radius = self.align_f0(f0, ctx)
            else:
                radius = self.radius
                if "rotary" in self.debug:
                    print(f"{layer} radius: {radius} ctx: {ctx}")
        else:
            radius = freqs
            
        freqs = torch.polar(torch.ones_like(radius), freqs.unsqueeze(0))
        
        self._counter += 1
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
