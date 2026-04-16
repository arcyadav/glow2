import os
import math
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from model import Glow


# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Hyperparameters
# -------------------------------
class HPS:
    in_channels = 1
    n_levels = 3
    depth = 8
    width = 128


hps = HPS()

batch_size = 64
epochs = 10
lr = 1e-4
temperature = 0.7

logdir = "./logs"
ckptdir = os.path.join(logdir, "checkpoints")
sampledir = os.path.join(logdir, "samples")
os.makedirs(logdir, exist_ok=True)
os.makedirs(ckptdir, exist_ok=True)
os.makedirs(sampledir, exist_ok=True)


# -------------------------------
# Data
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),   # [0, 1]
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)


# -------------------------------
# Model + Optimizer
# -------------------------------
model = Glow(hps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# -------------------------------
# Dequantization
# -------------------------------
def dequantize(x):
    # x assumed in [0, 1]
    x = x * 255.0
    x = x + torch.rand_like(x)
    x = x / 256.0
    return x


# -------------------------------
# Standard Gaussian log prob
# -------------------------------
def gaussian_log_p(z):
    return (-0.5 * (z ** 2 + math.log(2.0 * math.pi))).sum(dim=[1, 2, 3])


# -------------------------------
# Loss: Bits Per Dimension
# -------------------------------
def compute_loss(z_list, logdet, x):
    log_p = torch.zeros(x.size(0), device=x.device)

    for z in z_list:
        log_p = log_p + gaussian_log_p(z)

    log_px = log_p + logdet
    nll = -log_px

    num_dims = x.size(1) * x.size(2) * x.size(3)
    bpd = nll / (math.log(2.0) * num_dims)

    return bpd.mean(), log_p.mean(), logdet.mean()


# -------------------------------
# Sampling
# -------------------------------
@torch.no_grad()
def sample(model, n=16, temperature=0.7):
    model.eval()

    z_list = []
    for shape in model.z_shapes:
        B, C, H, W = shape
        z = torch.randn(n, C, H, W, device=device) * temperature
        z_list.append(z)

    x = model.reverse(z_list)
    x = torch.clamp(x, 0.0, 1.0)
    return x


# -------------------------------
# Warmup forward pass
# -------------------------------
print("Running warmup forward pass...")
x_init, _ = next(iter(loader))
x_init = x_init.to(device)
x_init = dequantize(x_init)

with torch.no_grad():
    _ = model(x_init)


# -------------------------------
# Training
# -------------------------------
print(f"Starting training on {device}...")

for epoch in range(1, epochs + 1):
    model.train()
    total_bpd = 0.0
    total_logp = 0.0
    total_logdet = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for x, _ in pbar:
        x = x.to(device)
        x = dequantize(x)

        z_list, logdet = model(x)
        loss, mean_logp, mean_logdet = compute_loss(z_list, logdet, x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_bpd += loss.item()
        total_logp += mean_logp.item()
        total_logdet += mean_logdet.item()

        pbar.set_postfix(
            bpd=f"{loss.item():.4f}",
            logp=f"{mean_logp.item():.1f}",
            logdet=f"{mean_logdet.item():.1f}"
        )

    avg_bpd = total_bpd / len(loader)
    avg_logp = total_logp / len(loader)
    avg_logdet = total_logdet / len(loader)

    print(
        f"Epoch {epoch} | "
        f"Avg BPD: {avg_bpd:.4f} | "
        f"Avg log_p(z): {avg_logp:.2f} | "
        f"Avg logdet: {avg_logdet:.2f}"
    )

    # -------------------------------
    # Save samples every epoch
    # -------------------------------
    samples = sample(model, n=16, temperature=temperature)
    save_image(samples, os.path.join(sampledir, f"samples_epoch_{epoch:03d}.png"), nrow=4)

    # -------------------------------
    # Save checkpoint every epoch
    # -------------------------------
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "z_shapes": model.z_shapes,
            "hps": {
                "in_channels": hps.in_channels,
                "n_levels": hps.n_levels,
                "depth": hps.depth,
                "width": hps.width,
            },
        },
        os.path.join(ckptdir, f"glow_epoch_{epoch:03d}.pt")
    )

print("Training complete!")