
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
        inv_freq = (theta / 140.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
        self.inv_freq = nn.Parameter(torch.tensor(inv_freq, device=device, dtype=dtype), requires_grad=True)

    def update_base(self, f0):
        f0 = f0.squeeze(0).to(device, dtype)
        theta = f0.mean() + 1e-8
        inv_freq = (theta / 140.0) * 700 * (torch.pow(10, torch.linspace(0, 2595 * torch.log10(torch.tensor(1 + 8000/700)), self.dim // 2, device=device, dtype=dtype) / 2595) - 1) / 1000
        self.inv_freq.data.copy_(inv_freq)
        self.theta.data.copy_(theta)    

    def return_f0(self, f0=None):
        if f0 is not None:
            self.f0 = f0
            return f0.squeeze(0).to(device, dtype)
        elif hasattr(self, 'f0') and self.f0 is not None:
            return self.f0.squeeze(0).to(device, dtype)
        return None

    def get_pitch_bias(self, f0):
        if f0 is None:
            return None
        f0_flat = f0.squeeze().float()
        f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
        f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
                                    f0_norm.unsqueeze(1)) * self.pitch_scale)
        return f0_sim.unsqueeze(0).unsqueeze(0)

    def f0proj(self, f0):
        if f0.ndim == 3:
            f0 = f0.squeeze(0)
        self.f0_proj = nn.Linear(1, self.head_dim // 2, device=device, dtype=dtype)
        f0 = f0.to(device, dtype)
        f0 = self.f0_proj(f0.unsqueeze(-1))  
        if f0.ndim == 3:
            f0 = f0.squeeze(0)
        return f0.to(device=device, dtype=dtype)

    def align_f0(self, ctx):
        f0 = self.return_f0()
        f0 = self.f0proj(f0)
        if f0.dim() == 3:
            batch, length, dims = f0.shape
            if length == ctx:
                return f0
            frames = length / ctx
            idx = torch.arange(ctx, device=f0.device)
            idx = (idx * frames).long().clamp(0, length - 1)
            return f0[:, idx, :]
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
       
    def forward(self, x=None, f0=None, enc=None, layer=None, input_type="audio") -> Tensor:
        if isinstance(x, int):
            ctx = x
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            batch, ctx, dims = x.shape
        else:
            batch, head, ctx, head_dim = x.shape
        t = torch.arange(ctx, device=device, dtype=dtype)
        freqs = self.inv_freq
        freqs = t[:, None] * freqs[None, :]

        if self.radii:
            radius = self.align_f0(ctx)
            if "rotary2" in self.debug:
                print(f"{layer} radius: {radius} ctx: {ctx}")
        else:
            radius = freqs
        freqs = torch.polar(torch.ones_like(radius), freqs)

        if "rotary3" in self.debug:
            print(f"{layer} count {self._counter} f0: {f0.shape if f0 is not None else None} freqs: {freqs.shape}  radius: {radius.shape} ctx: {ctx}")
            print(f"freqs mean: {freqs.mean():.2f} inv_freq mean: {self.inv_freq.mean():.2f} theta: {self.theta.item():.2f} radius mean: {radius.mean():.2f} radius shape: {radius.shape} ctx: {ctx}")

        if "rotary_detail" in self.debug:
            print(f"\n==== Detailed RoPE Analysis ====")
            print(f"Layer: {layer}, Context Length: {ctx}")
            print(f"F0 stats: mean={self.theta.item():.2f}")
            print(f"inv_freq range: [{self.inv_freq.min().item():.4f}, {self.inv_freq.max().item():.4f}]")
            
            if self.radii:
                print(f"Radius Shape: {radius.shape}, Mean: {radius.mean().item():.4f}")
                print(f"Radius[0]: {radius[0][:5].cpu().numpy()}") 
                print(f"Radius[mid]: {radius[ctx//2][:5].cpu().numpy()}")  
                print(f"Radius[end]: {radius[-1][:5].cpu().numpy()}")  
            
            print(f"Final freqs shape: {freqs.shape}")
            print(f"Freqs[0]: {freqs[0][:5].cpu().numpy()}")
            print(f"Freqs[mid]: {freqs[ctx//2][:5].cpu().numpy()}")
            print(f"Freqs[end]: {freqs[-1][:5].cpu().numpy()}")
            print("================================\n")      
        
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
