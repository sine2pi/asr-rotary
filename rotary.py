
class rotary(nn.Module):
    def __init__(self, dims, head, max_ctx=1500, theta=10000, radii=False, debug: List[str] = [], 
                 use_pbias=False, spec_shape=None):
        super().__init__()

        self.use_pbias = use_pbias
        self.last_f0_theta = None
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.max_ctx = max_ctx
        self.radii = radii
        self.learned_adaptation: bool = False
        radius = 1
        dim = self.head_dim
        self.dim = dim

        theta = torch.tensor(theta, device=device, dtype=dtype)
        self.theta = nn.Parameter(torch.tensor(theta, device=device, dtype=dtype), requires_grad=True)
        self.radius = nn.Parameter(torch.ones(radius, device=device, dtype=dtype), requires_grad=True)
        inv_freq = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
        self.inv_freq = nn.Parameter(torch.tensor(inv_freq, device=device, dtype=dtype), requires_grad=True)

    def update_base(self, f0):
        f0 = f0.squeeze(0).to(device, dtype)
        theta = f0.mean() + 1e-8
        inv_freq = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
        self.inv_freq.data.copy_(inv_freq)
        self.theta.data.copy_(theta)

    def get_pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)) * self.pitch_scale)
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def f0proj(self, f0):
            self.f0_proj = nn.Linear(1, self.head_dim // 2, device=device, dtype=dtype)
            f0 = f0.to(device, dtype)
            f0 = self.f0_proj(f0.unsqueeze(-1))  
            return f0.to(device=device, dtype=dtype) 

    def align_f0(self, f0, ctx):
        f0 = self.f0proj(f0)
        print(f"Aligning f0 with context: {ctx}, f0 shape: {f0}")
        if f0.dim() == 1:
            length = f0.shape[0]
            if length == ctx:
                return f0
            frames = length / ctx
            idx = torch.arange(ctx, device=f0.device)
            idx = (idx * frames).long().clamp(0, length - 1)
            return f0[idx]
        else:
            length, dims = f0.shape
            if length == ctx:
                return f0 
            frames = length / ctx
            idx = torch.arange(ctx, device=f0.device)
            idx = (idx * frames).long().clamp(0, length - 1)
            return f0[idx, :]

    def forward(self, x=None, enc=None, layer=None, input_type="audio") -> Tensor:
        f0 = enc.get("f0", None) if enc is not None else None

        if isinstance(x, int):
            ctx = x
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            batch, ctx, dims = x.shape
        else:
            batch, head, ctx, head_dim = x.shape
            
        t = torch.arange(ctx, device=device, dtype=dtype)

        if f0 is not None:
            freqs = self.inv_freq
            f0_mean = f0.mean()
            theta = f0_mean + 1e-8
            freqs = (theta / 220.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000

            if "rotary1" in self.debug:
                print(f"{layer}: {theta:.2f} : {f0_mean:.2f} : {ctx} ")
        else:
            freqs = self.inv_freq
        freqs = t[:, None] * freqs[None, :]
        if self.radii:
            if f0 is not None:
                radius = self.align_f0(f0, ctx)
            else:
                radius = freqs
                if "rotary2" in self.debug:
                    print(f"{layer} radius: {radius} ctx: {ctx}")
        else:
            radius = freqs
        freqs = torch.polar(torch.ones_like(radius), freqs.unsqueeze(0))

        if "rotary3" in self.debug:
            print(f"{layer} radius: {f0.shape if f0 is not None else None} ctx: {ctx}")

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
