import torch
import torch.nn as nn


# -------------------------------
# Polyak Averaging (EMA)
# -------------------------------

class PolyakEMA:
    def __init__(self, model, beta):
        self.beta = beta
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.beta * self.shadow[name]
                    + (1.0 - self.beta) * param.data
                )

    def swap(self, model):
        # Swap params <-> EMA params (like TF version)
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


# -------------------------------
# Custom Adam (Glow-style)
# -------------------------------

class GlowAdam:
    def __init__(self, model, hps, lr=3e-4, eps=1e-8):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.hps = hps

        self.params = [p for p in model.parameters() if p.requires_grad]

        self.beta1 = hps.beta1
        self.beta2 = 1 - 1.0 / (hps.train_its * hps.polyak_epochs)

        self.m1 = [torch.zeros_like(p) for p in self.params]
        self.m2 = [torch.zeros_like(p) for p in self.params]

        self.t = 1

        self.ema = PolyakEMA(model, self.beta2)

    def step(self):
        alpha_t = self.lr * (
            (1 - self.beta2 ** self.t) ** 0.5
            / (1 - self.beta1 ** self.t)
        )

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Momentum 1
            if self.beta1 > 0:
                self.m1[i] = self.beta1 * self.m1[i] + (1 - self.beta1) * g
                m1 = self.m1[i]
            else:
                m1 = g

            # Momentum 2
            self.m2[i] = self.beta2 * self.m2[i] + (1 - self.beta2) * (g ** 2)

            # Update
            delta = m1 / (torch.sqrt(self.m2[i]) + self.eps)

            p.data = self.hps.weight_decay * p.data - alpha_t * delta

        self.t += 1

        # Polyak update
        self.ema.update(self.model)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# -------------------------------
# Adamax variant
# -------------------------------

class GlowAdamax:
    def __init__(self, model, hps, lr=3e-4, eps=1e-8):
        self.model = model
        self.lr = lr
        self.eps = eps
        self.hps = hps

        self.params = [p for p in model.parameters() if p.requires_grad]

        self.beta1 = hps.beta1
        self.beta2 = 1 - 1.0 / (hps.train_its * hps.polyak_epochs)

        self.m1 = [torch.zeros_like(p) for p in self.params]
        self.m2 = [torch.zeros_like(p) for p in self.params]

        self.t = 1

        self.ema = PolyakEMA(model, self.beta2)

    def step(self):
        alpha_t = self.lr * (
            (1 - self.beta2 ** self.t) ** 0.5
            / (1 - self.beta1 ** self.t)
        )

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            if self.beta1 > 0:
                self.m1[i] = self.beta1 * self.m1[i] + (1 - self.beta1) * g
                m1 = self.m1[i]
            else:
                m1 = g

            self.m2[i] = torch.maximum(self.beta2 * self.m2[i], torch.abs(g))

            delta = m1 / (self.m2[i] + self.eps)

            p.data = self.hps.weight_decay * p.data - alpha_t * delta

        self.t += 1
        self.ema.update(self.model)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


# -------------------------------
# Factory (matches TF API)
# -------------------------------

def get_optimizer(model, hps, lr):
    if hps.optimizer == "adam":
        return GlowAdam(model, hps, lr)
    elif hps.optimizer == "adamax":
        return GlowAdamax(model, hps, lr)
    else:
        raise ValueError("Unknown optimizer")