import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -------------------------------
# Shape utils
# -------------------------------

def int_shape(x):
    return list(x.shape)


# -------------------------------
# ActNorm (already in model, but keeping consistent)
# -------------------------------

class ActNorm(nn.Module):
    def __init__(self, num_channels, logscale_factor=3.0):
        super().__init__()
        self.initialized = False
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.logscale_factor = logscale_factor

    def initialize(self, x):
        with torch.no_grad():
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            var = ((x - mean) ** 2).mean(dim=[0, 2, 3], keepdim=True)

            self.bias.data = -mean
            self.logs.data = torch.log(1.0 / (torch.sqrt(var) + 1e-6))

        self.initialized = True

    def forward(self, x, logdet=None, reverse=False):
        if not self.initialized:
            self.initialize(x)

        logs = self.logs * self.logscale_factor

        if not reverse:
            x = (x + self.bias) * torch.exp(logs)
        else:
            x = x * torch.exp(-logs) - self.bias

        if logdet is not None:
            dlogdet = torch.sum(logs) * x.shape[2] * x.shape[3]
            if reverse:
                dlogdet = -dlogdet
            logdet = logdet + dlogdet
            return x, logdet

        return x


# -------------------------------
# Linear layers
# -------------------------------

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, actnorm=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.actnorm = ActNorm(out_dim) if actnorm else None

    def forward(self, x):
        x = self.linear(x)
        if self.actnorm:
            x = self.actnorm(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return x


class LinearZeros(nn.Module):
    def __init__(self, in_dim, out_dim, logscale_factor=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.logs = nn.Parameter(torch.zeros(1, out_dim))
        self.logscale_factor = logscale_factor

    def forward(self, x):
        x = self.linear(x)
        x = x * torch.exp(self.logs * self.logscale_factor)
        return x


# -------------------------------
# Conv layers
# -------------------------------

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, actnorm=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.actnorm = ActNorm(out_channels) if actnorm else None

    def forward(self, x):
        x = self.conv(x)
        if self.actnorm:
            x = self.actnorm(x)
        return x


class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, logscale_factor=3):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.logscale_factor = logscale_factor

    def forward(self, x):
        x = self.conv(x)
        x = x * torch.exp(self.logs * self.logscale_factor)
        return x


# -------------------------------
# Feature permutations
# -------------------------------

def reverse_features(x):
    return torch.flip(x, dims=[1])


class ShuffleFeatures(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        indices = np.arange(num_channels)
        np.random.shuffle(indices)

        inv_indices = np.argsort(indices)

        self.register_buffer("indices", torch.tensor(indices))
        self.register_buffer("inv_indices", torch.tensor(inv_indices))

    def forward(self, x, reverse=False):
        if not reverse:
            return x[:, self.indices]
        else:
            return x[:, self.inv_indices]


# -------------------------------
# Gaussian distribution
# -------------------------------

class GaussianDiag:
    def __init__(self, mean, logsd):
        self.mean = mean
        self.logsd = logsd

    def sample(self):
        eps = torch.randn_like(self.mean)
        return self.mean + torch.exp(self.logsd) * eps

    def sample2(self, eps):
        return self.mean + torch.exp(self.logsd) * eps

    def logps(self, x):
        return -0.5 * (
            np.log(2 * np.pi)
            + 2.0 * self.logsd
            + (x - self.mean) ** 2 / torch.exp(2.0 * self.logsd)
        )

    def logp(self, x):
        return self.logps(x).view(x.size(0), -1).sum(dim=1)

    def get_eps(self, x):
        return (x - self.mean) / torch.exp(self.logsd)


# -------------------------------
# Flatten sum
# -------------------------------

def flatten_sum(x):
    return x.view(x.size(0), -1).sum(dim=1)


# -------------------------------
# Squeeze / Unsqueeze
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