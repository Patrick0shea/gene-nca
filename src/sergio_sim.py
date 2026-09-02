"""
sergio_sim.py — a from-scratch reimplementation of the core SERGIO algorithm
(Dibaeinia & Sinha, 2020, *Cell Systems*, "SERGIO: A Single-Cell Expression
Simulator Guided by Gene Regulatory Networks") for generating richer, more
realistic synthetic single-cell expression data from a known GRN.

Why reimplemented rather than imported: the real SERGIO is distributed only
via GitHub (github.com/PayamDiba/SERGIO), not on PyPI, and this environment's
network policy does not allow fetching arbitrary GitHub repos. So this module
re-derives SERGIO's published model straight from the paper instead of
depending on an unreachable package. It is a genuine upgrade over grn.py's
toy simulator, not a renamed copy of it:

  - Kinetics: each target gene's production rate is a sum of Hill-function
    contributions from its regulators (an increasing Hill curve for an
    activator, a decreasing one for a repressor) instead of the toy
    simulator's single global sigmoid of a linear combination. Master
    regulators get a fixed basal production rate instead of a regulatory
    input, exactly as in SERGIO.
  - Stochasticity: the population is simulated with the Chemical Langevin
    Equation (CLE) — a proper birth/death SDE with production- and
    decay-driven noise terms — integrated per CELL. Every simulated cell
    gets its own noisy trajectory from the SAME kinetic parameters, i.e.
    genuine cell-to-cell variability. The toy simulator was a single
    deterministic ODE with no notion of a "cell" at all.
  - Technical noise: optional dropout + library-size resampling, mimicking
    real scRNA-seq measurement on top of the biological signal (see
    `add_technical_noise`). Exposed for diagnostics; the model is still
    trained against the biological population mean (pseudobulk), matching
    the "one settled state ≈ pseudobulk" framing in the README.

A knockout is modelled exactly as in grn.py: clamp the perturbed gene's
concentration to a fixed value in every cell at every step of the CLE
integration, and let the coupled system re-settle around it. Because
production rates are recomputed from live regulator concentrations every
step, the clamp genuinely ripples through the Hill functions of downstream
genes — nothing about a knockout is special-cased per gene.

This module mirrors grn.py's public shape (make_grn / regulators) so it's a
drop-in alternative backend; see data.py for the branch point.
"""
from __future__ import annotations
import numpy as np
from . import grn as grnmod

regulators = grnmod.regulators  # same logic: any gene with >=1 outgoing edge


def make_grn(n_genes: int = 60,
             n_master: int = 5,
             avg_in_degree: float = 3.0,
             repress_frac: float = 0.4,
             hill_n: float = 2.0,
             k_half: float = 1.0,
             decay: float = 0.8,
             k_scale: float = 1.4,
             seed: int = 0):
    """Build a signed, directed GRN parameterised for Hill-kinetics simulation.

    Returns a dict with:
      K   : (n,n) interaction strength. K[i,j] = strength of edge j -> i.
      sign: (n,n) +1 (activates) / -1 (represses) per edge.
      b   : (n,) basal production rate — nonzero ONLY for master regulators.
      lam : (n,) decay rate per gene.
      A   : (n,n) binary structure (A[i,j]=1 if j regulates i) — the only
            network info the model is allowed to see.
    """
    rng = np.random.default_rng(seed)
    is_master = np.zeros(n_genes, dtype=bool)
    masters = rng.choice(n_genes, size=n_master, replace=False)
    is_master[masters] = True

    out_budget = np.full(n_genes, 2)
    out_budget[masters] = rng.integers(8, 16, size=n_master)

    K = np.zeros((n_genes, n_genes))
    sign = np.zeros((n_genes, n_genes))
    total_edges = int(avg_in_degree * n_genes)
    made = 0
    guard = 0
    while made < total_edges and guard < total_edges * 50:
        guard += 1
        p = out_budget / out_budget.sum()
        j = rng.choice(n_genes, p=p)             # regulator (source)
        i = rng.integers(n_genes)                 # target
        if i == j or K[i, j] != 0 or is_master[i]:  # masters take no input
            continue
        s = -1.0 if rng.random() < repress_frac else 1.0
        mag = rng.gamma(2.0, 0.5) * k_scale / np.sqrt(avg_in_degree)
        K[i, j] = mag
        sign[i, j] = s
        out_budget[j] = max(1, out_budget[j] - 1)
        made += 1

    b = np.zeros(n_genes)
    b[masters] = rng.uniform(0.8, 1.6, size=n_master)
    lam = np.full(n_genes, decay)
    A = (K != 0).astype(np.float32)
    return {"K": K, "sign": sign, "b": b, "lam": lam, "A": A,
            "masters": masters, "is_master": is_master,
            "hill_n": hill_n, "k_half": k_half, "n_genes": n_genes}


def _hill_active(x, k_half, n):
    xn = np.power(x, n)
    return xn / (k_half ** n + xn)


def _hill_repress(x, k_half, n):
    kn = k_half ** n
    return kn / (kn + np.power(x, n))


def simulate(grn, n_cells: int = 200, steps: int = 400, dt: float = 0.05,
             clamp: dict | None = None, seed: int = 0, burn_frac: float = 0.3):
    """Simulate `n_cells` independent cells via the Chemical Langevin Equation.

    Returns (X_final, pseudobulk):
      X_final    : (n_cells, n_genes) per-cell state at the last step — this
                   IS the single-cell realism payoff, e.g. figures/single_cell
                   uses it to show cell-to-cell spread.
      pseudobulk : (n_genes,) population mean over the post-burn-in trailing
                   window — this is the ground truth the model is trained on
                   (the biological steady state, i.e. pseudobulk).
    """
    rng = np.random.default_rng(seed)
    n = grn["n_genes"]
    K_pos = np.where(grn["sign"] > 0, grn["K"], 0.0)
    K_neg = np.where(grn["sign"] < 0, grn["K"], 0.0)
    n_hill, k_half = grn["hill_n"], grn["k_half"]
    lam, b = grn["lam"], grn["b"]

    X = np.zeros((n_cells, n))
    clamp_idx = clamp_val = None
    if clamp:
        clamp_idx = np.array(list(clamp.keys()), dtype=int)
        clamp_val = np.array(list(clamp.values()), dtype=float)
        X[:, clamp_idx] = clamp_val

    burn = int(steps * (1 - burn_frac))
    traj_sum = np.zeros(n)
    n_avg = 0
    for t in range(steps):
        Ha = _hill_active(X, k_half, n_hill)
        Hr = _hill_repress(X, k_half, n_hill)
        P = Ha @ K_pos.T + Hr @ K_neg.T + b[None, :]   # (n_cells, n_genes)

        dW1 = rng.standard_normal(X.shape)
        dW2 = rng.standard_normal(X.shape)
        drift = (P - lam[None, :] * X) * dt
        diffusion = (np.sqrt(np.clip(P * dt, 0, None)) * dW1
                     - np.sqrt(np.clip(lam[None, :] * X * dt, 0, None)) * dW2)
        X = np.clip(X + drift + diffusion, 0.0, None)

        if clamp_idx is not None:
            X[:, clamp_idx] = clamp_val
        if t >= burn:
            traj_sum += X.mean(axis=0)
            n_avg += 1

    pseudobulk = traj_sum / max(n_avg, 1)
    return X, pseudobulk


def generate_dataset(grn, genes_to_perturb, n_cells: int = 200, steps: int = 400,
                      dt: float = 0.05, knockout_value: float = 0.0, seed: int = 0):
    """Simulate the control population and every requested single-gene
    knockout, then rescale everything by ONE shared factor so the numeric
    range matches the toy simulator's (~[0, 1]) — this is what lets model.py,
    train.py, evaluate.py and visualize.py (deg-threshold, clamp value, etc.)
    work unchanged regardless of which simulator built the dataset.

    Returns (control, control_cells, targets):
      control       : (n_genes,) normalized pseudobulk of the unperturbed population
      control_cells : (n_cells, n_genes) normalized per-cell control snapshot
      targets       : {gene: (n_genes,) normalized knockout pseudobulk}
    """
    control_cells_raw, control_raw = simulate(
        grn, n_cells=n_cells, steps=steps, dt=dt, clamp=None, seed=seed)

    targets_raw = {}
    for k in genes_to_perturb:
        k = int(k)
        _, pb = simulate(grn, n_cells=n_cells, steps=steps, dt=dt,
                         clamp={k: knockout_value}, seed=seed + 1 + k)
        targets_raw[k] = pb

    scale = max(control_raw.max(),
                max((v.max() for v in targets_raw.values()), default=0.0),
                1e-6) * 1.15

    control = (control_raw / scale).astype(np.float32)
    control_cells = (control_cells_raw / scale).astype(np.float32)
    targets = {k: (v / scale).astype(np.float32) for k, v in targets_raw.items()}
    return control, control_cells, targets


def add_technical_noise(X, dropout_shape: float = 8.0, dropout_mid: float = 0.3,
                         lib_size_cv: float = 0.15, seed: int = 0):
    """Layer SERGIO-style technical noise on top of biological signal `X`
    (n_cells, n_genes), purely for diagnostics — NOT used as a training
    target. Two effects real scRNA-seq protocols add on top of biology:

      dropout    : low-expressed transcripts are stochastically missed. We
                   use a logistic dropout probability that falls as
                   expression rises (Dibaeinia & Sinha's model).
      library size: each cell is sequenced to a different total depth, so
                   apparent counts are that cell's true signal times a
                   per-cell log-normal scale factor.
    """
    rng = np.random.default_rng(seed)
    p_dropout = 1.0 / (1.0 + np.exp(dropout_shape * (X - dropout_mid)))
    keep = rng.random(X.shape) > p_dropout
    lib_factor = rng.lognormal(mean=0.0, sigma=lib_size_cv, size=(X.shape[0], 1))
    return X * keep * lib_factor, p_dropout.mean()


if __name__ == "__main__":
    g = make_grn(seed=1)
    reg = regulators(g)
    control, control_cells, targets = generate_dataset(g, reg[:3], seed=1)
    delta = targets[int(reg[0])] - control
    print(f"genes={g['n_genes']} regulators={len(reg)} masters={g['masters']}")
    print(f"control mean={control.mean():.3f}  "
          f"per-cell control std (avg over genes)={control_cells.std(axis=0).mean():.4f}")
    print(f"KO of gene {reg[0]}: {np.sum(np.abs(delta) > 0.05)} genes moved >0.05")
    noisy, drop_rate = add_technical_noise(control_cells)
    print(f"technical dropout rate: {drop_rate:.2%}")
