# class rotary(nn.Module):
#     _seen = set()  
#     def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, radii=False,
#                  learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = []):
#         super().__init__()
#         self.use_pbias = False

#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.device = device
#         dtype = torch.float32 
#         self.dtype = dtype
#         self.debug = debug
#         self._counter = 0
#         self.dims = dims
#         self.max_ctx = max_ctx
#         self.radii = radii
        
#         self.inv_freq = nn.Parameter(
#                 1.0 / (10000 ** (torch.arange(0, dims, 2, device=device, dtype=dtype) / dims)),
#                 requires_grad=learned_freq)
#         self.theta = nn.Parameter(
#             torch.tensor(float(theta)), requires_grad=learned_theta)
#         self.min_theta = nn.Parameter(
#             torch.tensor(600.0), requires_grad=learned_theta)
#         self.max_theta = nn.Parameter(
#             torch.tensor(2400.0), requires_grad=learned_theta)
        
#         self.pitch_scale = nn.Parameter(torch.tensor(1.0), 
#                                         requires_grad=learned_pitch)
    
#         if radii:
#             self.radius = nn.Parameter(
#                 torch.ones(dims // 2),
#                 requires_grad=learned_radius)

#     def get_pitch_bias(self, f0):
#         if f0 is None:
#             return None
            
#         f0_flat = f0.squeeze().float()
#         f0_norm = (f0_flat - f0_flat.mean()) / (f0_flat.std() + 1e-8)
#         f0_sim = torch.exp(-torch.cdist(f0_norm.unsqueeze(1), 
#                                     f0_norm.unsqueeze(1)) * self.pitch_scale)
#         return f0_sim.unsqueeze(0).unsqueeze(0)

#     def add_to_rotary(self):
#         def get_sim(self, freqs):
#             real = freqs.real.squeeze(0)
#             imag = freqs.imag.squeeze(0)
#             vecs = torch.cat([real.unsqueeze(-2), imag.unsqueeze(-2)], dim=-1)
#             vecs = vecs.squeeze(-2)
#             return F.cosine_similarity(vecs.unsqueeze(1), vecs.unsqueeze(0), dim=-1)
            
#         def fwd_sim(self, x=None, f0=None):
#             freqs = self.forward(x, f0)
#             sim = get_sim(self, freqs)
#             return freqs, sim
            
#         rotary.get_sim = get_sim
#         rotary.fwd_sim = fwd_sim

#     def align_f0_to_tokens(self, f0, token_length):
#         ratio = len(f0) / token_length
#         indices = [int(i * ratio) for i in range(token_length)]
#         indices = [min(i, len(f0) - 1) for i in indices]
#         return f0[indices]

#     def forward(self, x=None, f0=None, stage=None) -> Tensor:
#         if isinstance(x, int):
#             t = torch.arange(x, device=self.device).float()
#         else:
#             t = x.float().to(self.inv_freq.device)

#         if f0 is not None:
#             # f0_mean = f0.mean()
#             f0_mean = f0.mean()
#             f0_theta = self.theta + (1.0 / f0_mean) * self.theta
#             # f0_mean = torch.clamp(f0_m2, min=80.0, max=600.0)
#             # pf = torch.log(1 + f0_mean / 700.0) / torch.log(torch.tensor(1 + 300.0 / 700.0))
#             # f0_theta = self.min_theta + pf * (self.max_theta - self.min_theta)
#             # print(f"f0_m2: {f0_m2} Hz, F0: {f0_m}, Theta: {f0_theta} Hz")
#             inv_freq = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=self.device) / self.dims))
#         else:
#             inv_freq = self.inv_freq
#         freqs = torch.einsum('i,j->ij', t, inv_freq)

#         freqs = freqs.float()
#         if self.radii:

# # if stage == 'decoder' and f0 is not None:
# #       f0 = self.align_f0_to_tokens(f0, freqs.shape[-1])
# #           radius = f0
#             radius = F.softplus(self.radius)
# #           freqs = torch.polar(radius, freqs)
#             freqs = torch.polar(radius.unsqueeze(0).expand_as(freqs), freqs)
#         else:
#             freqs = torch.polar(torch.ones_like(freqs), freqs)
#         freqs = freqs.unsqueeze(0)
            
#         if "rotary" in self.debug:
#             if f0 is not None:
#                 key = f"{self._counter}_{f0_theta:.2f}"
#                 if key not in rotary._seen:
#                     if not hasattr(self, '_prev_f0_theta'):
#                         self._prev_f0_theta = f0_theta
#                         print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
#                     elif abs(self._prev_f0_theta - f0_theta) > 1000.0:
#                         print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
#                         self._prev_f0_theta = f0_theta
#                     rotary._seen.add(key)
#             self._counter += 1
#         return freqs      

#     @staticmethod
#     def apply_rotary(x, freqs):
#         multihead_format = len(freqs.shape) == 4
#         if multihead_format:
#             x1 = x[..., :freqs.shape[-1]*2]
#             x2 = x[..., freqs.shape[-1]*2:]
#             x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
#             x1 = torch.view_as_complex(x1)
#             x1 = x1 * freqs
#             x1 = torch.view_as_real(x1).flatten(-2)
#             return torch.cat([x1.type_as(x), x2], dim=-1)
#         else:
#             x1 = x[..., :freqs.shape[-1]*2]
#             x2 = x[..., freqs.shape[-1]*2:]
            
#             if x.ndim == 2:  
#                 x1 = x1.unsqueeze(0)
#                 x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
#                 x1 = torch.view_as_complex(x1)
#                 x1 = x1 * freqs
#                 x1 = torch.view_as_real(x1).flatten(-2)
#                 x1 = x1.squeeze(0)  
#                 return torch.cat([x1.type_as(x), x2], dim=-1)
#             else:  
#                 x1 = x1.float().reshape(*x1.shape[:-1], -1, 2).contiguous()
#                 x1 = torch.view_as_complex(x1)
#                 x1 = x1 * freqs
#                 x1 = torch.view_as_real(x1).flatten(-2)
#                 return torch.cat([x1.type_as(x), x2], dim=-1)


class rotary(nn.Module):
    _seen = set()  
    def __init__(self, dims, max_ctx=1500, theta=10000, learned_freq=False, radii=False,
                 learned_radius=False, learned_theta=False, learned_pitch=False, debug: List[str] = []):
        super().__init__()
        self.use_pbias = False

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        dtype = torch.float32 
        self.dtype = dtype
        self.debug = debug
        self._counter = 0
        self.dims = dims
        self.max_ctx = max_ctx
        self.radii = radii
        pitch_scale = 1.0 
        # theta_rescale = 1.0
        # theta *= theta_rescale ** (dims / (dims - 2))

        self.min_theta = nn.Parameter(
            torch.tensor(20.0), requires_grad=learned_theta)
        self.max_theta = nn.Parameter(
            torch.tensor(400.0), requires_grad=learned_theta)

        self.theta = nn.Parameter(
            torch.tensor(float(theta)), requires_grad=learned_theta)
        
        self.pitch_scale = nn.Parameter(torch.tensor(pitch_scale),
                                        requires_grad=learned_pitch)
    
        freqs = 1. / (theta ** (torch.arange(0, dims, 2)[:(dims // 2)].float() / dims))
        self.freqs = nn.Parameter(freqs, requires_grad = learned_freq)

        if radii:
            self.radius = nn.Parameter(torch.ones(dims // 2),
                requires_grad=learned_radius)

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

    def align_f0(self, f0, token_length):
        batch_size, f0_length = f0.shape
        if f0_length == token_length:
            return f0 # No resampling needed (encoder path - audio features)
        frames_per_token = f0_length / token_length
        
        indices = torch.arange(token_length, device=f0.device)
        indices = (indices * frames_per_token).long()#.clamp(max=f0_length-1)
        #center_positions = ((indices + 0.5) * frames_per_token).long()
        batch_indices = torch.arange(batch_size, device=f0.device).unsqueeze(1)
        return f0[batch_indices, indices.unsqueeze(0).expand(batch_size, -1)]
    
    def scale_f0(self, f0):
        f0_min = f0.min(dim=1, keepdim=True)[0]
        f0_max = f0.max(dim=1, keepdim=True)[0]
        denom = f0_max - f0_min + 1e-8
        normalized_f0 = (f0 - f0_min) / denom
        # normalized_f0 = (f0 - f0_min) / (f0_max - f0_min)
        normalized_f0 = torch.clamp(normalized_f0, 0.0, 1.0)
        return normalized_f0

    def process_f0(f0, threshold=0.05):
        thresholded_f0 = torch.where(f0 < threshold, torch.zeros_like(f0), f0)
        return thresholded_f0

    def map_perceptual(self, f0_mean, theta=10000.0):
        if f0_mean >= theta:
            return torch.log(f0_mean / theta)
        else:
            return -torch.log(theta / f0_mean)

    def linear_map(self, freq, min_freq=40.0, max_freq=400.0, target_max=10000.0):
        mapped_freq = ((freq - min_freq) / (max_freq - min_freq)) * target_max
        return mapped_freq

    def log_map(self, freq, min_freq=40.0, max_freq=400.0, target_max=10000.0):
        log_freq = torch.log(freq)
        log_min_freq = torch.log(min_freq)
        log_max_freq = torch.log(max_freq)

        mapped_log_freq = ((log_freq - log_min_freq) / (log_max_freq - log_min_freq)) * torch.log(torch.tensor(target_max, device=self.device))
        return mapped_log_freq

    def forward(self, x=None, f0=None, stage=None) -> Tensor:
        if isinstance(x, int):
            seq_len = x
        else:
            batch, seq_len, _ = x.shape
        t = torch.arange(seq_len, device=self.device).float()

        if f0 is not None:
            f0_mean = f0.mean()
            theta = self.theta
            f0_theta = theta * (f0_mean * 1e-2 + 1.0)
            freqs = 1.0 / (f0_theta ** (torch.arange(0, self.dims, 2, device=self.device) / self.dims))
        else:        
            freqs = self.freqs
            
        freqs = torch.einsum('i,j->ij', t, freqs)
        freqs = freqs.float()
        
        if self.radii and f0 is not None:
            radius = self.align_f0(f0, seq_len)
            # radius = self.scale_f0(radius)
            radius = F.softplus(self.radius) * radius
            # radius = radius.unsqueeze(-1)  # Ensure radius is of shape (batch, seq_len, dims//2)
            freqs = torch.polar(radius.unsqueeze(-1), freqs.unsqueeze(0))
        else:
            freqs = torch.polar(torch.ones_like(freqs), freqs.unsqueeze(0))
        # print(f"Step {self._counter}:   Block: {stage}:   Radius: {radius}")
        if "rotary" in self.debug:
            if f0 is not None:
                key = f"{self._counter}_{f0_theta:.2f}"
                if key not in rotary._seen:
                    if not hasattr(self, '_prev_f0_theta'):
                        self._prev_f0_theta = f0_theta
                        print(f"Step {self._counter}: Using raw F0 as theta: {f0_theta:.2f} Hz")
                    elif abs(self._prev_f0_theta - f0_theta) > 1000.0:
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
