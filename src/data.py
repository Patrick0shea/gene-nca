"""
data.py — turn the synthetic GRN into a supervised perturbation dataset.

Each example is a single-gene knockout:
    input  : the control (unperturbed) expression state + which gene is clamped
    target : the ground-truth perturbed steady state (from grn.steady_state)

Crucially we split the *perturbed genes* into train and test. The model is
trained on knockouts of the train genes and must predict knockouts of genes it
has NEVER been perturbed on. That held-out generalization is the whole test:
it can only succeed by using the network structure to propagate a novel clamp,
not by memorizing.
"""
from __future__ import annotations
import numpy as np
from . import grn as grnmod


def build_dataset(n_genes=60, seed=0, knockout_value=0.0, test_frac=0.25):
    g = grnmod.make_grn(n_genes=n_genes, seed=seed)
    control = grnmod.steady_state(g)
    reg = grnmod.regulators(g)

    rng = np.random.default_rng(seed + 100)
    reg = rng.permutation(reg)
    n_test = max(4, int(len(reg) * test_frac))
    test_genes = np.sort(reg[:n_test])
    train_genes = np.sort(reg[n_test:])

    targets = grnmod.perturbation_targets(g, control, reg, knockout_value)

    return {
        "grn": g,
        "control": control.astype(np.float32),
        "targets": {int(k): v.astype(np.float32) for k, v in targets.items()},
        "train_genes": [int(x) for x in train_genes],
        "test_genes": [int(x) for x in test_genes],
        "knockout_value": knockout_value,
    }


def summarize(ds):
    ctrl = ds["control"]
    moved = []
    for k, tgt in ds["targets"].items():
        moved.append(int(np.sum(np.abs(tgt - ctrl) > 0.05)))
    moved = np.array(moved)
    print(f"genes={ds['grn']['n_genes']}  "
          f"perturbations={len(ds['targets'])} "
          f"(train={len(ds['train_genes'])}, test={len(ds['test_genes'])})")
    print(f"per-knockout genes moved >0.05: "
          f"min={moved.min()} median={int(np.median(moved))} max={moved.max()}")
