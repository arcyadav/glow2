import os
import torch
import torch.nn as nn
import numpy as np
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
    n_levels = 3
    depth = 8
    width = 128
    flow_permutation = 2
    flow_coupling = 1

hps = HPS()

batch_size = 64
epochs = 10
lr = 1e-4   # 🔥 FIXED (was too high)

logdir = "./logs"
os.makedirs(logdir, exist_ok=True)

# -------------------------------
# Data
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x - 0.5)
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# -------------------------------
# Model
# -------------------------------
model = Glow(hps).to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

# -------------------------------
# Loss (Bits Per Dimension)
# -------------------------------
def compute_loss(z_list, logdet, image_size=32, channels=3):
    log_p = 0

    for z in z_list:
        log_p += (-0.5 * (np.log(2 * np.pi) + z**2)).sum(dim=[1, 2, 3])

    log_likelihood = log_p + logdet
    nll = -log_likelihood

    # 🔥 IMPORTANT: Normalize → BPD
    bpd = nll / (np.log(2) * channels * image_size * image_size)

    return bpd.mean()

# -------------------------------
# Sampling
# -------------------------------
def sample(model, n=16, temperature=0.7):
    model.eval()
    with torch.no_grad():
        z_list = []

        for shape in model.z_shapes:
            z = torch.randn(n, shape[1], shape[2], shape[3]).to(device)
            z *= temperature
            z_list.append(z)

        x = model.reverse(z_list)
        x = x + 0.5  # back to [0,1]

    return x

# -------------------------------
# 🔥 Warmup (VERY IMPORTANT)
# -------------------------------
# Needed to initialize ActNorm + z_shapes
print("Running warmup forward pass...")
x_init, _ = next(iter(loader))
x_init = x_init.to(device)
x_init = x_init + torch.rand_like(x_init) / 256.0

with torch.no_grad():
    _ = model(x_init)

# -------------------------------
# Training
# -------------------------------
print(f"Starting training on {device}...")

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for x, _ in pbar:
        x = x.to(device)

        # Dequantization
        x = x + torch.rand_like(x) / 256.0

        # Forward
        z_list, logdet = model(x)

        loss = compute_loss(z_list, logdet)

        optimizer.zero_grad()
        loss.backward()

        # Stability
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(bpd=f"{loss.item():.3f}")

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} | Avg BPD: {avg_loss:.4f}")

    # -------------------------------
    # Sampling
    # -------------------------------
    samples = sample(model, n=16)
    samples = torch.clamp(samples, 0, 1)

    save_image(samples, os.path.join(logdir, f"samples_epoch_{epoch}.png"), nrow=4)

    # -------------------------------
    # Checkpoint
    # -------------------------------
    if epoch % 5 == 0:
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch
        }, os.path.join(logdir, f"glow_epoch_{epoch}.pt"))

print("Training complete!")