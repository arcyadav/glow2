import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------------
# Utilities
# -------------------------------
def squeeze2d(x, factor=2):
    if factor == 1:
        return x
    B, C, H, W = x.shape
    assert H % factor == 0 and W % factor == 0
    x = x.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(x, factor=2):
    if factor == 1:
        return x
    B, C, H, W = x.shape
    assert C % (factor ** 2) == 0
    x = x.view(B, C // (factor ** 2), factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor ** 2), H * factor, W * factor)
    return x


# -------------------------------
# ActNorm
# -------------------------------
class ActNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.initialized = False
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True, unbiased=False)
            std = torch.clamp(std, min=self.eps)

            self.bias.data.copy_(-mean)
            self.logs.data.copy_(torch.log(1.0 / std))

        self.initialized = True

    def forward(self, x, logdet=None, reverse=False):
        if not self.initialized:
            self.initialize(x)

        _, _, H, W = x.shape
        dlogdet = torch.sum(self.logs) * H * W

        if not reverse:
            x = (x + self.bias) * torch.exp(self.logs)
            if logdet is not None:
                logdet = logdet + dlogdet
        else:
            x = x * torch.exp(-self.logs) - self.bias
            if logdet is not None:
                logdet = logdet - dlogdet

        return x, logdet


# -------------------------------
# Invertible 1x1 Convolution
# -------------------------------
class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_init = np.linalg.qr(np.random.randn(num_channels, num_channels))[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, logdet=None, reverse=False):
        B, C, H, W = x.shape

        sign, logabsdet = torch.slogdet(self.weight.double())
        dlogdet = logabsdet.float() * H * W

        if not reverse:
            weight = self.weight.view(C, C, 1, 1)
            x = F.conv2d(x, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
        else:
            weight_inv = torch.inverse(self.weight.double()).float().view(C, C, 1, 1)
            x = F.conv2d(x, weight_inv)
            if logdet is not None:
                logdet = logdet - dlogdet

        return x, logdet


# -------------------------------
# Small NN for coupling
# -------------------------------
class NN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        # Identity initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


# -------------------------------
# Affine Coupling
# -------------------------------
class AffineCoupling(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        assert channels % 2 == 0
        self.channels = channels
        self.nn = NN(channels // 2, hidden_channels, channels)

        # learnable scale factor to keep s small initially
        self.scale_factor = nn.Parameter(torch.zeros(1, channels // 2, 1, 1))

    def forward(self, x, logdet=None, reverse=False):
        x1, x2 = x.chunk(2, dim=1)

        h = self.nn(x1)
        t, s = h.chunk(2, dim=1)

        # stable scaling
        s = torch.tanh(s) * torch.exp(self.scale_factor)

        if not reverse:
            x2 = x2 * torch.exp(s) + t
            if logdet is not None:
                logdet = logdet + s.sum(dim=[1, 2, 3])
        else:
            x2 = (x2 - t) * torch.exp(-s)
            if logdet is not None:
                logdet = logdet - s.sum(dim=[1, 2, 3])

        x = torch.cat([x1, x2], dim=1)
        return x, logdet


# -------------------------------
# One Glow step
# -------------------------------
class FlowStep(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.invconv = Invertible1x1Conv(channels)
        self.coupling = AffineCoupling(channels, hidden_channels)

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            x, logdet = self.actnorm(x, logdet, reverse=False)
            x, logdet = self.invconv(x, logdet, reverse=False)
            x, logdet = self.coupling(x, logdet, reverse=False)
        else:
            x, logdet = self.coupling(x, logdet, reverse=True)
            x, logdet = self.invconv(x, logdet, reverse=True)
            x, logdet = self.actnorm(x, logdet, reverse=True)
        return x, logdet


# -------------------------------
# Block / Level
# -------------------------------
class GlowLevel(nn.Module):
    def __init__(self, channels, depth, hidden_channels):
        super().__init__()
        self.steps = nn.ModuleList([
            FlowStep(channels, hidden_channels) for _ in range(depth)
        ])

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            for step in self.steps:
                x, logdet = step(x, logdet, reverse=False)
        else:
            for step in reversed(self.steps):
                x, logdet = step(x, logdet, reverse=True)
        return x, logdet


# -------------------------------
# Multi-scale Glow
# -------------------------------
class Glow(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.levels = nn.ModuleList()
        self.z_shapes = []

        C = getattr(hps, "in_channels", 1)

        for level_idx in range(hps.n_levels):
            C = C * 4
            self.levels.append(GlowLevel(C, hps.depth, hps.width))

            if level_idx < hps.n_levels - 1:
                C = C // 2

    def forward(self, x):
        logdet = torch.zeros(x.size(0), device=x.device)
        z_list = []
        self.z_shapes = []

        for i, level in enumerate(self.levels):
            x = squeeze2d(x, factor=2)
            x, logdet = level(x, logdet, reverse=False)

            if i < len(self.levels) - 1:
                x, z = x.chunk(2, dim=1)
                z_list.append(z)
                self.z_shapes.append(tuple(z.shape))

        z_list.append(x)
        self.z_shapes.append(tuple(x.shape))
        return z_list, logdet

    def reverse(self, z_list):
        x = z_list[-1]
        logdet = torch.zeros(x.size(0), device=x.device)

        for i in reversed(range(len(self.levels))):
            if i < len(self.levels) - 1:
                x = torch.cat([x, z_list[i]], dim=1)

            x, logdet = self.levels[i](x, logdet, reverse=True)
            x = unsqueeze2d(x, factor=2)

        return x