import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def hz_to_mel(f):  # optional; not used yet
    return 2595.0 * torch.log10(1 + f / 700.0)

class SincConv1d(nn.Module):
    """Sinc-based band-pass conv following SincNet (Mirco Ravanelli et al.).
    Kernel is constructed from learnable low/high cutoff frequencies.
    """
    def __init__(self, out_channels: int, kernel_size: int, sample_rate: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Initialize cutoff frequencies in Hz (avoid 0 and Nyquist)
        low_hz = 30.0
        high_hz = sample_rate / 2 - (low_hz + 1)
        hz = torch.linspace(low_hz, high_hz, out_channels + 1)
        self.f1 = nn.Parameter(hz[:-1].view(-1, 1))  # [F,1]
        self.f2 = nn.Parameter(hz[1:].view(-1, 1))   # [F,1]

        # Hamming window
        n = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1).float()
        self.register_buffer("n", n)

    def forward(self, x):
        # Enforce f1 < f2 and positivity
        f1 = torch.abs(self.f1)
        f2 = f1 + torch.abs(self.f2 - self.f1) + 1.0
        # Convert to angular frequencies
        min_freq = 1.0
        f1 = f1 + min_freq
        f2 = f2 + min_freq

        # Sinc band-pass = 2f2*sinc(2πf2t) - 2f1*sinc(2πf1t)
        n = self.n.to(x.device)
        t_right = n[n >= 0]
        filters = []
        for i in range(self.out_channels):
            f1i = f1[i, 0] / self.sample_rate
            f2i = f2[i, 0] / self.sample_rate
            # time-domain low/high-pass
            band = (2 * f2i) * torch.sinc(2 * math.pi * f2i * n) - (2 * f1i) * torch.sinc(2 * math.pi * f1i * n)
            # Hamming window
            window = 0.54 - 0.46 * torch.cos(2 * math.pi * (torch.arange(self.kernel_size, device=x.device) / (self.kernel_size - 1)))
            band = band * window - band.mean()
            filters.append(band)
        filters = torch.stack(filters).unsqueeze(1)  # [F,1,K]
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, groups=1)

class SincPDNet(nn.Module):
    def __init__(self, n_channels: int, input_len: int, n_filters: int = 16, kernel_size: int = 129, sample_rate: float = 100.0, task: str = "classification"):
        super().__init__()
        self.front = nn.Sequential(
            SincConv1d(n_filters, kernel_size, sample_rate),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        # A small CNN head
        self.conv = nn.Sequential(
            nn.Conv1d(n_filters, n_filters*2, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        feat_dim = (n_filters * 2) * 32
        if task == "regression":
            self.head = nn.Linear(feat_dim, 1)
            self.out_act = None
            self.is_reg = True
        else:
            self.head = nn.Linear(feat_dim, 1)
            self.out_act = nn.Sigmoid()
            self.is_reg = False

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2).contiguous()
        x = self.front(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        if self.out_act is not None:
            x = self.out_act(x)
        return x.squeeze(-1)
