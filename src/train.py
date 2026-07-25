"""
train.py — train the GraphNCA on the training-gene knockouts.

We unroll the T settling steps and backpropagate through all of them
(backprop-through-time). The loss is a weighted MSE on the expression channel
that up-weights genes that actually move, so the model is pushed to get the
differentially-expressed genes right rather than just predicting "no change".
"""
from __future__ import annotations
import numpy as np
import torch


def _weighted_mse(pred, target, control, deg_weight=6.0):
    delta_true = torch.abs(target - control)
    w = 1.0 + deg_weight * delta_true          # emphasise genes that moved
    return torch.mean(w * (pred - target) ** 2)


def train(model, ds, steps=25, epochs=250, lr=3e-3, deg_weight=6.0,
          seed=0, verbose=True):
    torch.manual_seed(seed)
    control = torch.tensor(ds["control"])
    kv = ds["knockout_value"]
    train_genes = ds["train_genes"]
    targets = {k: torch.tensor(v) for k, v in ds["targets"].items()}

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    rng = np.random.default_rng(seed)
    history = []
    for ep in range(epochs):
        model.train()
        order = rng.permutation(train_genes)
        ep_loss = 0.0
        for k in order:
            k = int(k)
            pred = model(control, k, kv, steps=steps)
            loss = _weighted_mse(pred, targets[k], control, deg_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        ep_loss /= len(train_genes)
        history.append(ep_loss)
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  train weighted-MSE {ep_loss:.5f}")
    return history
