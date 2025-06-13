class rotary(nn.Module):
    _seen = set()  
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, variable_radius=False,
                 learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = []):
        super().__init__()
        
        self.dims = dims
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float32
        self.device = device
        self.dtype = dtype
        self.debug = debug
        self._counter = 0

        self.use_pbias = False    
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius
        
        self.inv_freq = nn.Parameter(1.0 / (theta ** (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)),
                requires_grad=learned_freq)

        self.theta = nn.Parameter(torch.tensor(float(theta)), 
                requires_grad=learned_theta)

        self.pitch_scale = nn.Parameter(torch.tensor(1.0), requires_grad=learned_pitch)
    
        if variable_radius:
            self.radius = nn.Parameter(torch.ones(dims // 2), requires_grad=learned_radius)

    def get_pitch_bias(self, f0):
        if f0 is None:
            return None
            
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)) * self.pitch_scale)
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def add_to_rotary(self):
        def get_sim(self, freqs):
            real = freqs.real.squeeze(0)
            imag = freqs.imag.squeeze(0)
            vecs = torch.cat([real.unsqueeze(-2), imag.unsqueeze(-2)], dim=-1)
            vecs = vecs.squeeze(-2)
            return F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
            
        def fwd_sim(self, x=None, f0=None):
            freqs = self.forward(x, f0)
            sim = get_sim(self, freqs)
            return freqs, sim
            
        rotary.get_sim = get_sim
        rotary.fwd_sim = fwd_sim

    def align_f0_to_tokens(self, f0, token_length):
        ratio = len(f0) / token_length
        indices = [int(i * ratio) for i in range(token_length)]
        indices = [min(i, len(f0) - 1) for i in indices]
        return f0[indices]

    def forward(self, x=None, f0=None, stage=None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=self.device).float()
        else:
            t = x.float().to(self.inv_freq.device)
        if f0 is not None:
            f0_mean = f0.mean()
            f0_theta = (f0_mean**2) * self.pitch_scale
            #f0_theta = f0_mean * (f0_mean / self.theta) * self.theta * self.pitch_scale
            inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=self.device) / self.dims)) 
        else:
            inv_freq = self.inv_freq
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        freqs = freqs.float()
        if self.variable_radius:
            if f0 is not None:
                f0 = f0[0]
                seq_len = x
                f0 = torch.tensor(f0, device=device if isinstance(x, torch.Tensor) else device)
                f0 = self.align_f0_to_tokens(f0, freqs.shape[-1])
                radius = 1.0 / (f0 + 1)
                freqs = torch.polar(radius, freqs)
            else:
                freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)

        if "rotary" in self.debug:
            if f0 is not None:
                key = f"{self._counter}_{f0_theta:.2f}"
                if key not in rotary._seen:
                    if not hasattr(self, '_prev_f0_theta'):
                        self._prev_f0_theta = f0_theta
                        print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
                    elif abs(self._prev_f0_theta - f0_theta) > 0.0:
                        print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
                        self._prev_f0_theta = f0_theta
                    rotary._seen.add(key)
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
            
            if x.ndim == 2:  
                x1 = x1.unsqueeze(0)
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs
                x1 = torch.view_as_real(x1).flatten(-2)
                x1 = x1.squeeze(0)  
                return torch.cat([x1.type_as(x), x2], dim=-1)
            else:  
                x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
                x1 = torch.view_as_complex(x1)
                x1 = x1 * freqs
                x1 = torch.view_as_real(x1).flatten(-2)
                return torch.cat([x1.type_as(x), x2], dim=-1)
