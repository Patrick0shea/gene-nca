"""
grn.py — the sandbox's ground truth.

We build a synthetic gene-regulatory network (GRN) whose wiring WE control, and
define a cell's expression state as the *steady state* (attractor) of a simple
regulatory dynamics on that network. A perturbation is a clamp: pin one gene's
expression, let the network re-settle, and the new steady state is the ground
truth we grade our model against.

Nothing here is meant to be biologically exact. It is a transparent test rig:
because we know the true wiring, we know the true answer for every perturbation,
so we can tell whether the model actually learned to propagate effects — before
we ever touch messy real data. (Graduate this to SERGIO / real Perturb-seq later.)
"""
from __future__ import annotations
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_grn(n_genes: int = 60,
             n_master: int = 5,
             avg_in_degree: float = 3.0,
             repress_frac: float = 0.4,
             weight_scale: float = 1.6,
             seed: int = 0):
    """Build a signed, directed GRN.

    Returns a dict with:
      W  : (n,n) signed weight matrix. W[i,j] = effect of gene j ON gene i.
      b  : (n,) basal drive per gene (its default tendency without regulators).
      A  : (n,n) binary structure (A[i,j]=1 if j regulates i). This is the ONLY
           network info the model is allowed to see — not the signs or weights.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n_genes, n_genes))

    # A few "master regulators" get many outgoing edges (broad influence);
    # everyone else gets a handful. This gives perturbations a realistic spread
    # of impact — knocking out a master ripples widely, a leaf barely at all.
    masters = rng.choice(n_genes, size=n_master, replace=False)
    out_budget = np.full(n_genes, 2)
    out_budget[masters] = rng.integers(8, 16, size=n_master)

    total_edges = int(avg_in_degree * n_genes)
    made = 0
    guard = 0
    while made < total_edges and guard < total_edges * 50:
        guard += 1
        # pick a source weighted by remaining out-budget (masters dominate)
        p = out_budget / out_budget.sum()
        j = rng.choice(n_genes, p=p)            # regulator (source)
        i = rng.integers(n_genes)               # target
        if i == j or W[i, j] != 0:
            continue
        sign = -1.0 if rng.random() < repress_frac else 1.0
        mag = rng.gamma(2.0, 0.5) * weight_scale / np.sqrt(avg_in_degree)
        W[i, j] = sign * mag
        out_budget[j] = max(1, out_budget[j] - 1)
        made += 1

    b = rng.normal(0.0, 0.75, size=n_genes)     # per-gene basal drive
    A = (W != 0).astype(np.float32)
    return {"W": W, "b": b, "A": A, "masters": masters, "n_genes": n_genes}


def steady_state(grn, clamp: dict | None = None,
                 iters: int = 400, damping: float = 0.5, tol: float = 1e-8):
    """Iterate x <- (1-d)x + d*sigmoid(b + W x) to the network's fixed point.

    clamp: {gene_index: value} held fixed every step (this is how we model a
    knockout: clamp the gene's expression to 0 and let everything else re-settle).
    """
    W, b, n = grn["W"], grn["b"], grn["n_genes"]
    x = np.full(n, 0.5)
    if clamp:
        for k, v in clamp.items():
            x[k] = v
    for _ in range(iters):
        z = b + W @ x
        x_new = (1 - damping) * x + damping * _sigmoid(z)
        if clamp:
            for k, v in clamp.items():
                x_new[k] = v
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        x = x_new
    return x


def perturbation_targets(grn, control, genes, knockout_value: float = 0.0):
    """Ground-truth perturbed steady states for a list of single-gene knockouts."""
    targets = {}
    for k in genes:
        targets[k] = steady_state(grn, clamp={int(k): knockout_value})
    return targets


def regulators(grn):
    """Genes with at least one outgoing edge — the only ones worth perturbing,
    since knocking out a gene that regulates nothing produces no ripple."""
    out_deg = (grn["A"].sum(axis=0) > 0)
    return np.where(out_deg)[0]


if __name__ == "__main__":
    g = make_grn(seed=1)
    ctrl = steady_state(g)
    reg = regulators(g)
    ko = steady_state(g, clamp={int(reg[0]): 0.0})
    delta = ko - ctrl
    print(f"genes={g['n_genes']} regulators={len(reg)} masters={g['masters']}")
    print(f"control mean={ctrl.mean():.3f}  KO of gene {reg[0]}: "
          f"{np.sum(np.abs(delta) > 0.05)} genes moved >0.05")
