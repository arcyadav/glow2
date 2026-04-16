import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import Glow  # your model.py

# -------------------------------
# Device
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

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
lr = 1e-3
logdir = "./logs"
os.makedirs(logdir, exist_ok=True)

# -------------------------------
# Data (MNIST → 3x32x32)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# Model
# -------------------------------
model = Glow(hps).to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

# -------------------------------
# Loss
# -------------------------------
def gaussian_log_p(z):
    return -0.5 * (np.log(2 * np.pi) + z**2)

def compute_loss(z_list, logdet):
    log_p = 0
    for z in z_list:
        log_p += gaussian_log_p(z).sum(dim=[1,2,3])
    loss = -(log_p + logdet)
    return loss.mean()

# -------------------------------
# Sampling (approximate shapes)
# -------------------------------
def sample(model, n=16, temperature=0.7):
    z_list = []

    for shape in model.z_shapes:
        z = torch.randn(n, shape[1], shape[2], shape[3]).to(device)
        z *= temperature
        z_list.append(z)

    x = model.reverse(z_list)
    return x

# -------------------------------
# Checkpoint
# -------------------------------
def save_checkpoint(model, optimizer, epoch):
    path = os.path.join(logdir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }, path)

def load_checkpoint(path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]

# -------------------------------
# Training
# -------------------------------
print("Starting training...")

from tqdm import tqdm

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0

    # 🔥 Progress bar here
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

    for x, _ in pbar:
        x = x.to(device)

        z_list, logdet = model(x)
        loss = compute_loss(z_list, logdet)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 show live loss in bar
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

    # -------------------------------
    # Sampling
    # -------------------------------
    model.eval()
    with torch.no_grad():
        samples = sample(model, n=16)
        samples = torch.clamp(samples, 0, 1)
        save_image(samples, os.path.join(logdir, f"samples_epoch_{epoch}.png"), nrow=4)

    # -------------------------------
    # Save checkpoint
    # -------------------------------
    if epoch % 5 == 0:
        save_checkpoint(model, optimizer, epoch)

print("Training complete!")