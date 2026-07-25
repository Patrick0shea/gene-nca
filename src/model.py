"""
model.py — GraphNCA.

A neural cellular automaton whose grid has been replaced by the gene-regulatory
graph. One small shared rule is applied to every gene, every step:

    1. each gene reads its neighbours (regulators upstream, targets downstream)
    2. it combines that with its own state through a tiny shared MLP
    3. it nudges its own state by a small residual:  h <- h + f(h, neighbours)

Run the rule for T steps and the state settles. To predict a knockout we pin
the perturbed gene's expression channel to its silenced value every step and
let the rest of the network relax around it — the settled expression is the
prediction.

What the model is given: only the BINARY graph structure (who regulates whom).
It is NOT given the true signs, weights, basal drives, or the generator's
equations — it must learn the rule from the training perturbations alone.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def _row_normalize(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return A / deg


class GraphNCA(nn.Module):
    def __init__(self, structure: np.ndarray, channels: int = 16,
                 msg_dim: int = 16, hidden: int = 64):
        super().__init__()
        n = structure.shape[0]
        self.n = n
        self.channels = channels

        # Incoming = a gene's regulators (j -> i). Outgoing = genes it regulates.
        A_in = structure.astype(np.float32)          # A_in[i,j]=1 if j regulates i
        A_out = structure.T.astype(np.float32)        # A_out[i,j]=1 if i regulates j
        self.register_buffer("A_in", torch.tensor(_row_normalize(A_in)))
        self.register_buffer("A_out", torch.tensor(_row_normalize(A_out)))

        # learned transforms of neighbour states before aggregation
        self.lin_in = nn.Linear(channels, msg_dim)
        self.lin_out = nn.Linear(channels, msg_dim)

        # the shared local update rule
        self.rule = nn.Sequential(
            nn.Linear(channels + 2 * msg_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        # zero-init the last layer so early updates are ~0 (NCA stability trick)
        nn.init.zeros_(self.rule[-1].weight)
        nn.init.zeros_(self.rule[-1].bias)

    def _clamp_mask(self, clamp_idx, device):
        """(n, channels) mask that is 1 only at [clamp_idx, 0] (the expression
        channel of the perturbed gene)."""
        m = torch.zeros(self.n, self.channels, device=device)
        m[clamp_idx, 0] = 1.0
        return m

    def forward(self, control_expr, clamp_idx, clamp_val, steps=25,
                return_traj=False):
        device = self.lin_in.weight.device
        if not torch.is_tensor(control_expr):
            control_expr = torch.tensor(np.asarray(control_expr, dtype=np.float32))
        control_expr = control_expr.to(device)

        # initial state: expression channel = control, hidden channels = 0
        H = torch.zeros(self.n, self.channels, device=device)
        H = H + torch.nn.functional.pad(control_expr.unsqueeze(1),
                                        (0, self.channels - 1))

        mask = self._clamp_mask(clamp_idx, device)
        clamp_field = mask * clamp_val
        H = H * (1 - mask) + clamp_field           # apply the knockout clamp

        traj = [H[:, 0].detach().cpu().numpy()] if return_traj else None
        for _ in range(steps):
            m_in = self.A_in @ self.lin_in(H)      # mean of regulators' messages
            m_out = self.A_out @ self.lin_out(H)   # mean of downstream messages
            percept = torch.cat([H, m_in, m_out], dim=1)
            H = H + self.rule(percept)             # residual update
            H = H * (1 - mask) + clamp_field       # re-clamp every step
            if return_traj:
                traj.append(H[:, 0].detach().cpu().numpy())

        pred_expr = H[:, 0]
        if return_traj:
            return pred_expr, np.stack(traj)
        return pred_expr

    @torch.no_grad()
    def predict(self, control_expr, clamp_idx, clamp_val, steps=25):
        self.eval()
        return self.forward(control_expr, clamp_idx, clamp_val, steps).cpu().numpy()
