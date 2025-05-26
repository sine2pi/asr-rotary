
class rotary(nn.Module):
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, variable_radius=False,
                 learned_radius=False, debug=False):
        super().__init__()
        self.debug = False
        self.interpolate_factor = 10.0
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.variable_radius = variable_radius
        self.inv_freq = nn.Parameter(1.0 / (10000 ** (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)), requires_grad=learned_freq)
        if variable_radius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=learned_radius)
            
        self.theta = nn.Parameter(torch.tensor(float(theta)), requires_grad=False)

    def forward(self, x = None, f0=None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=device).float()
        else:
            t = x.float().to(self.inv_freq.device)

        if f0 is not None:
            f0_tensor = f0.squeeze(0) if f0.ndim == 3 else f0
            if f0_tensor.ndim > 1:
                f0_tensor = f0_tensor.squeeze()
            f0_mean = f0_tensor.mean()
            f0_mean = torch.clamp(f0_mean, min=80.0, max=600.0)
            perceptual_factor = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
            min_theta, max_theta = 800.0, 10000.0
            f0_theta = min_theta + perceptual_factor * (max_theta - min_theta)
            inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=device) / self.dims))
        else:
            inv_freq = self.inv_freq
        freqs = torch.einsum('i,j->ij', t, inv_freq)

        freqs = freqs.float()
        if self.variable_radius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)
        if self.debug:
            if self._counter == 1:
                print(f'ROTA -- freqs: {freqs.shape}, x: {x},  {t.shape if x is not None else None}', freqs.shape, t.shape)
            if f0 is not None and self._counter % 100 == 0:
                print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
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
