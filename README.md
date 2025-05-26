
Extract frequencies from raw audio and map 1:1 to rotary frequency during training. Works.

Intuition:

Different pitch regions should have different positional encoding characteristics
Higher pitch → faster rotations → positions become more distinct
Lower pitch → slower rotations → positions blend more smoothly

Linguistic: Pitch patterns in speech carry meaningful prosodic information (questions, emphasis, emotion). By making positional encodings F0-aware, we help the model distinguish these patterns.

Position-Specific Processing: Using different thetas for different positions based on their F0. 
This allows the model to:

Process high-pitched regions differently from low-pitched regions
Pay attention to pitch transitions
Potentially capture speech prosody better than fixed positional encodings

Replace the standard fixed theta with F0-based theta:

In high-pitched regions: faster rotations → positions become more distinct
In low-pitched regions: slower rotations → positions blend more smoothly
This effectively creates a "pitch-adaptive attention field" where the model's attention patterns dynamically adjust based on pitch characteristics.

Extract pitch:

``` python

def extract_features(batch, tokenizer, spectrogram=True, waveforms=True, pitch=True, f0=True, energy_contour=False, periodocity=False,
                     hop_length=128, fmin=0, fmax=8000, n_mels=128, n_fft=1024, sampling_rate=16000, pad_value=0.0,
                     pad_mode="constant", center=True, power=2.0, window_fn=torch.hann_window, mel_scale="htk", 
                     norm=None, normalized=False, debug=False):
    
    global model, extractor

    dtype = torch.float32
    device = torch.device("cuda:0")
    audio = batch["audio"]
    sampling_rate = audio["sampling_rate"]
        
    wav = torch.tensor(audio["array"]).float()
    sr = audio["sampling_rate"]
    
    if sr != sampling_rate:
        original_length = wav.shape[-1]
        target_length = int(original_length * (sampling_rate / sr))
        
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = resampler(wav)
        
        if abs(wav.shape[-1] - target_length) > 1:
            new_waveform = torch.zeros((wav.shape[0], target_length), dtype=dtype, device=device)
            copy_length = min(wav.shape[1], target_length)
            new_waveform[:, :copy_length] = wav[:, :copy_length]
            wav = new_waveform

    if spectrogram:
        transform = torchaudio.transforms.MelSpectrogram(
            f_max=fmax,
            f_min=fmin,
            n_mels=n_mels,
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            norm='slaney',
            normalized=False,
            power=2.0,
            center=True, 
            mel_scale="htk",
            window_fn=torch.hann_window,  
            )
    
        mel_spectrogram = transform(wav)      
        log_mel = torch.clamp(mel_spectrogram, min=1e-10).log10()
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        spec = (log_mel + 4.0) / 4.0
        spec = torch.tensor(spec)

    wav = wav.unsqueeze(0)

    if pitch:
        pit = torchcrepe.predict(
            wav, 
            sampling_rate, 
            hop_length,
            fmin=150,
            fmax=600,
            model="tiny",
            decoder=torchcrepe.decode.viterbi,
            return_periodicity=False, 
            device=device, 
            pad=False
        )
        
    if waveforms:
        batch["waveform"] = wav
    if pitch:
        batch["pitch"] = pit
    if spectrogram:
        batch["spectrogram"] = spec
    batch["labels"] = tokenizer.encode(batch["transcription"], add_special_tokens=False)
    return batch


```

Pass to specialized rotary:

```python


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

```
Pass through multihead as f0

```python


class MultiheadA(nn.Module):
    def __init__(self, dims: int, head: int, debug=False):
        super().__init__()

        self.count = 0
        self.debug = debug
        self.pad_token = 0
        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.q = Linear(dims, dims)
        self.k = Linear(dims, dims, bias=False)
        self.v = Linear(dims, dims)
        self.o = Linear(dims, dims)
        self.fzero = nn.Parameter(torch.tensor(0.0001))
        
        self.rope = rotary(
            dims=self.head_dim,
            max_ctx = 1500,
            theta = 10000,
            learned_freq = False,
            variable_radius = False,
            learned_radius = False,
            debug = False)

    def forward(self, x: Tensor, xa = None, mask = None, return_attn=False, f0=None):

        z = default(xa, x)
        q = self.q(x)
        k = self.k(z)
        v = self.v(z)

        if f0 is not None:
            qf = self.rope(q.size(1), f0=f0)
            kf = self.rope(k.size(1), f0=f0)
        else:
            qf = self.rope(q.size(1))
            kf = self.rope(k.size(1))

        bat, ctx, dims = q.shape
        scale = (dims // self.head) ** -0.25
        q = q.view(*q.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.head, -1).permute(0, 2, 1, 3)
        q = self.rope.apply_rotary(q, qf)
        k = self.rope.apply_rotary(k, kf)
        qk = (q * scale) @ (k * scale).transpose(-1, -2)
        token_ids = k[:, :, :, 0]
        zscale = torch.ones_like(token_ids)
        fzero = torch.clamp(F.softplus(self.fzero), min=0.00001, max=0.001)
        zscale[token_ids.float() == self.pad_token] = fzero.to(q.device, q.dtype)
        if mask is not None:
            mask = mask[:ctx, :ctx]
            qk = qk + mask.unsqueeze(0).unsqueeze(0) * zscale.unsqueeze(-2).expand(qk.shape)
        qk = qk * zscale.unsqueeze(-2)
        if return_attn:
            return qk, v
        w = F.softmax(qk, dim=-1).to(q.dtype)
        wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
        if self.debug and self.count % 100 == 0:
            print(f"Step {self.count}: x: {x.shape}, xa: {xa.shape if xa is not None else None}, mask: {mask.shape if mask is not None else None}")
        self.count += 1
        return self.o(wv), qk.detach()

```
