import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------------
# Utilities
# -------------------------------
def squeeze2d(x, factor=2):
    B, C, H, W = x.shape
    x = x.view(B, C, H//factor, factor, W//factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4)
    return x.reshape(B, C * factor * factor, H//factor, W//factor)


def unsqueeze2d(x, factor=2):
    B, C, H, W = x.shape
    x = x.view(B, C // (factor**2), factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3)
    return x.reshape(B, C // (factor**2), H * factor, W * factor)


# -------------------------------
# ActNorm
# -------------------------------
class ActNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.initialized = False
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            self.bias.data = -mean
            self.logs.data = torch.log(1 / (std + 1e-6))
        self.initialized = True

    def forward(self, x, logdet, reverse=False):
        if not self.initialized:
            self.initialize(x)

        B, C, H, W = x.shape

        if not reverse:
            x = (x + self.bias) * torch.exp(self.logs)
            logdet = logdet + torch.sum(self.logs) * H * W
        else:
            x = x * torch.exp(-self.logs) - self.bias
            logdet = logdet - torch.sum(self.logs) * H * W

        return x, logdet


# -------------------------------
# Invertible 1x1 Conv
# -------------------------------
class Invertible1x1Conv(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w = np.linalg.qr(np.random.randn(num_channels, num_channels))[0].astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(w))

    def forward(self, x, logdet, reverse=False):
        B, C, H, W = x.shape
        dlogdet = torch.slogdet(self.weight)[1] * H * W

        if not reverse:
            weight = self.weight.view(C, C, 1, 1)
            x = F.conv2d(x, weight)
            logdet = logdet + dlogdet
        else:
            weight_inv = torch.inverse(self.weight).view(C, C, 1, 1)
            x = F.conv2d(x, weight_inv)
            logdet = logdet - dlogdet

        return x, logdet


# -------------------------------
# Coupling NN
# -------------------------------
class NN(nn.Module):
    def __init__(self, in_channels, width, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, out_channels, 3, padding=1)
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return self.net(x)


# -------------------------------
# FlowStep
# -------------------------------
class FlowStep(nn.Module):
    def __init__(self, channels, hps):
        super().__init__()
        self.actnorm = ActNorm(channels)
        self.permutation = hps.flow_permutation
        self.coupling = hps.flow_coupling

        if self.permutation == 2:
            self.invconv = Invertible1x1Conv(channels)

        self.nn = NN(channels // 2, hps.width,
                     channels if self.coupling == 1 else channels // 2)

    def forward(self, x, logdet, reverse=False):

        if not reverse:
            x, logdet = self.actnorm(x, logdet)

            if self.permutation == 2:
                x, logdet = self.invconv(x, logdet)

            z1, z2 = x.chunk(2, dim=1)

            if self.coupling == 0:
                z2 = z2 + self.nn(z1)
            else:
                h = self.nn(z1)
                shift, scale = h.chunk(2, dim=1)
                scale = torch.sigmoid(scale + 2.)
                z2 = (z2 + shift) * scale
                logdet = logdet + torch.sum(torch.log(scale + 1e-6), dim=[1,2,3])

            x = torch.cat([z1, z2], dim=1)

        else:
            z1, z2 = x.chunk(2, dim=1)

            if self.coupling == 0:
                z2 = z2 - self.nn(z1)
            else:
                h = self.nn(z1)
                shift, scale = h.chunk(2, dim=1)
                scale = torch.sigmoid(scale + 2.)
                z2 = z2 / scale - shift
                logdet = logdet - torch.sum(torch.log(scale + 1e-6), dim=[1,2,3])

            x = torch.cat([z1, z2], dim=1)

            if self.permutation == 2:
                x, logdet = self.invconv(x, logdet, reverse=True)

            x, logdet = self.actnorm(x, logdet, reverse=True)

        return x, logdet


# -------------------------------
# RevNet
# -------------------------------
class RevNet(nn.Module):
    def __init__(self, channels, hps):
        super().__init__()
        self.steps = nn.ModuleList([
            FlowStep(channels, hps) for _ in range(hps.depth)
        ])

    def forward(self, x, logdet, reverse=False):
        if not reverse:
            for step in self.steps:
                x, logdet = step(x, logdet)
        else:
            for step in reversed(self.steps):
                x, logdet = step(x, logdet, reverse=True)
        return x, logdet


# -------------------------------
# Glow
# -------------------------------
class Glow(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.levels = nn.ModuleList()

        C = 3
        for i in range(hps.n_levels):
            C = C * 4  # after squeeze
            self.levels.append(RevNet(C, hps))

            if i < hps.n_levels - 1:
                C = C // 2  # after split

    # -------------------
    # FORWARD (encode)
    # -------------------
    def forward(self, x, reverse=False, eps_list=None):

        if not reverse:
            logdet = torch.zeros(x.size(0), device=x.device)
            z_list = []

            for i, level in enumerate(self.levels):
                x = squeeze2d(x)  # ✅ per level squeeze

                x, logdet = level(x, logdet)

                if i < len(self.levels) - 1:
                    x, z = x.chunk(2, dim=1)
                    z_list.append(z)

            z_list.append(x)

            # store shapes for sampling
            self.z_shapes = [z.shape for z in z_list]

            return z_list, logdet

        else:
            return self.reverse(eps_list)

    # -------------------
    # REVERSE (decode)
    # -------------------
    def reverse(self, z_list):
        assert z_list is not None

        x = z_list[-1]
        logdet = torch.zeros(x.size(0), device=x.device)

        for i in reversed(range(len(self.levels))):

            if i < len(self.levels) - 1:
                x = torch.cat([x, z_list[i]], dim=1)

            x, logdet = self.levels[i](x, logdet, reverse=True)

            x = unsqueeze2d(x)  # ✅ per level unsqueeze

        return x