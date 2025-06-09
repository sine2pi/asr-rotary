
class rotary(nn.Module):
    _seen = set()  
    def __init__(self, dims, max_ctx=1500, theta=10000, lfreq=True, vradius=False,
                 lradius=False, ltheta=True, lpitch=True, debug: List[str] = []):
        super().__init__()
        self.use_pbias = False
        self.last_f0_theta = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        dtype = torch.float32 
        self.dtype = dtype
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.vradius = vradius
        self.cp = nn.Linear(512, 512, bias=False)  
        self.btw_g = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.inv_freq = nn.Parameter(
                1.0 / (10000 ** (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)),
                requires_grad=lfreq)
        self.theta = nn.Parameter(
            torch.tensor(float(theta)), requires_grad=ltheta)      
        self.pitch_scale = nn.Parameter(torch.tensor(1.0), 
                                        requires_grad=lpitch)
        if vradius:
            self.radius = nn.Parameter(
                torch.ones(dims // 2),
                requires_grad=lradius)

    def pbias(self, f0):
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
                
    def get_btw(self, x):
        orig_shape = x.shape
        orig_dim = len(orig_shape)
        
        if orig_dim == 4:
            b, s, h, d = orig_shape
            x_flat = x.transpose(1, 2).reshape(b * h, s, d)
        else:
            b, s, d = orig_shape
            h = 1
            x_flat = x

        if not hasattr(self, '_cp_dim') or self._cp_dim != d:
            self.cp = nn.Linear(d, d, bias=False).to(x.device, x.dtype)
            self._cp_dim = d
                
        c = self.cp(x_flat)
        ti = c.unsqueeze(2)
        tj = c.unsqueeze(1)
        dist = torch.norm(ti - tj, dim=-1)
        btw = torch.zeros(x_flat.shape[0], s, device=x.device)
            
        if s > 2:
            src = torch.arange(s-2, device=x.device)
            mid = src + 1
            tgt = src + 2
            
            batch = x_flat.shape[0]
            fbatch = torch.arange(batch, device=x.device).repeat_interleave(s-2)
            fsrc = src.repeat(batch)
            fmid = mid.repeat(batch)
            ftgt = tgt.repeat(batch)
            
            direct = dist[fbatch, fsrc, ftgt].view(batch, s-2)
            path1 = dist[fbatch, fsrc, fmid].view(batch, s-2)
            path2 = dist[fbatch, fmid, ftgt].view(batch, s-2)
            
            path = path1 + path2
            scores = torch.relu(1.0 - (path - direct) / torch.clamp(direct, min=1e-6))
            btw = torch.zeros(batch, s, device=x.device)
            btw[:, 1:s-1] = scores
            btw = btw / (s - 2)
        
        if orig_dim == 4:
            btw = btw.view(b, h, s).transpose(1, 2)
        elif orig_dim == 3:
            btw = btw.unsqueeze(-1)
        return btw
        
    def btw_rope(self, x, btw):
        original_dim = x.dim()
        original_shape = x.shape
        
        if original_dim == 3:
            b, s, d = x.shape
            h = 1
            x = x.view(b, s, h, d)
        else:
            b, s, h, d = x.shape
        dev = x.device
        
        t = torch.arange(self.max_ctx, device=dev).float()
        frqs = self.forward(t)
        fc = frqs.real
        fs = frqs.imag
        rope_dim = fc.size(-1)
        pos = torch.arange(s, device=dev).view(1, s, 1, 1)
        
        if btw.size(1) != s:
            btw = self.get_btw(x)
        
        if btw.dim() == 3:
            if btw.size(2) == 1:
                btw = btw.expand(-1, -1, h)
            btw = btw.unsqueeze(-1)
        
        a = self.btw_g * (btw - 0.5) * 2.0
        pa = torch.clamp(pos + a, 0, self.max_ctx - 1)
        lo = torch.floor(pa).long()
        hi = torch.ceil(pa).long()
        f = pa - lo
        
        fc = fc.unsqueeze(2).expand(b, -1, h, -1)
        fs = fs.unsqueeze(2).expand(b, -1, h, -1)
        li = lo.expand(-1, -1, -1, rope_dim)
        hi = hi.expand(-1, -1, -1, rope_dim)
        cl = torch.gather(fc, 1, li)
        ch = torch.gather(fc, 1, hi)
        sl = torch.gather(fs, 1, li)
        sh = torch.gather(fs, 1, hi)
        
        c = (1-f)*cl + f*ch
        s = (1-f)*sl + f*sh
        
        repeat_factor = d // (2 * rope_dim)
        if repeat_factor > 1:
            c = c.repeat_interleave(repeat_factor, dim=-1)
            s = s.repeat_interleave(repeat_factor, dim=-1)
        
        xo = torch.zeros_like(x)
        xe = x[..., 0::2]
        xd = x[..., 1::2]
        xo[..., 0::2] = xe*c - xd*s
        xo[..., 1::2] = xd*c + xe*s
        
        if original_dim == 3:
            xo = xo.reshape(original_shape)
        return xo.type_as(x)
            
    def forward(self, x = None, f0=None) -> Tensor:
        if isinstance(x, int):
            t = torch.arange(x, device=self.device).float()
        else:
            t = x.float().to(self.inv_freq.device)
        if f0 is not None:
            f0_tensor = f0.squeeze(0) if f0.ndim == 3 else f0
            if f0_tensor.ndim > 1:
                f0_tensor = f0_tensor.squeeze()
            f0_mean = f0_tensor.mean()
            f0_mean = torch.clamp(f0_mean, min=80.0, max=600.0)
            perceptual = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 600.0 / 700.0))
            f0_theta = self.theta * (1.0 + perceptual)
            self.last_f0_theta = f0_theta.item()
            inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=self.device) / self.dims))
        else:
            inv_freq = self.inv_freq
        freqs = torch.einsum('i,j->ij', t, inv_freq)   
        freqs = freqs.float()
        if self.vradius:
            radius = F.softplus(self.radius)
            freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs)
        freqs = freqs.unsqueeze(0)
            
        if "rotary" in self.debug and self._counter % 100 == 0:
            if f0 is not None:
                key = f"{self._counter}_{f0_theta:.2f}"
                if key not in rotary._seen:
                    if not hasattr(self, '_prev_f0_theta'):
                        self._prev_f0_theta = f0_theta
                        print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
                    elif abs(self._prev_f0_theta - f0_theta) > 500.0:
                        print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
                        self._prev_f0_theta = f0_theta
                    print(f"Freqs shape: {freqs.shape}, Inv_freq shape: {inv_freq.shape}, F0: {f0_theta if f0 is not None else 'N/A'}")
                    rotary._seen.add(key)
            self._counter += 1
        return freqs      

    @staticmethod
    def apply_rotary(x, freqs):
        multihead = len(freqs.shape) == 4
        if multihead:
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
