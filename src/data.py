"""
data.py — turn a simulated GRN into a supervised perturbation dataset.

Each example is a single-gene knockout:
    input  : the control (unperturbed) expression state + which gene is clamped
    target : the ground-truth perturbed steady state

Crucially we split the *perturbed genes* into train and test. The model is
trained on knockouts of the train genes and must predict knockouts of genes it
has NEVER been perturbed on. That held-out generalization is the whole test:
it can only succeed by using the network structure to propagate a novel clamp,
not by memorizing.

Two interchangeable ground-truth simulators, selected by `simulator`:
  "sergio" (default) : src/sergio_sim.py — a from-scratch reimplementation of
                        SERGIO's Hill-kinetics + Chemical Langevin Equation
                        model. Genuinely single-cell: every knockout is a
                        population of independently-noisy cells, pseudobulked.
  "toy"               : src/grn.py — the original deterministic single-sigmoid
                        ODE. Kept as a fast, noise-free debugging rig (see
                        README "Honest caveats").
Either way the returned dict has the same shape, so model.py / train.py /
evaluate.py / visualize.py don't need to know which one built the data.
"""
from __future__ import annotations
import numpy as np
from . import grn as grnmod
from . import sergio_sim


def build_dataset(n_genes=60, seed=0, knockout_value=0.0, test_frac=0.25,
                   simulator="sergio", n_cells=200):
    if simulator == "toy":
        return _build_toy(n_genes, seed, knockout_value, test_frac)
    if simulator == "sergio":
        return _build_sergio(n_genes, seed, knockout_value, test_frac, n_cells)
    raise ValueError(f"unknown simulator {simulator!r}, expected 'sergio' or 'toy'")


def _split_genes(reg, seed, test_frac):
    rng = np.random.default_rng(seed + 100)
    reg = rng.permutation(reg)
    n_test = max(4, int(len(reg) * test_frac))
    return np.sort(reg[n_test:]), np.sort(reg[:n_test])   # train, test


def _build_toy(n_genes, seed, knockout_value, test_frac):
    g = grnmod.make_grn(n_genes=n_genes, seed=seed)
    control = grnmod.steady_state(g)
    reg = grnmod.regulators(g)
    train_genes, test_genes = _split_genes(reg, seed, test_frac)

    targets = grnmod.perturbation_targets(g, control, reg, knockout_value)

    return {
        "grn": g,
        "control": control.astype(np.float32),
        "targets": {int(k): v.astype(np.float32) for k, v in targets.items()},
        "train_genes": [int(x) for x in train_genes],
        "test_genes": [int(x) for x in test_genes],
        "knockout_value": knockout_value,
        "simulator": "toy",
    }


def _build_sergio(n_genes, seed, knockout_value, test_frac, n_cells):
    g = sergio_sim.make_grn(n_genes=n_genes, seed=seed)
    reg = sergio_sim.regulators(g)
    train_genes, test_genes = _split_genes(reg, seed, test_frac)

    control, control_cells, targets = sergio_sim.generate_dataset(
        g, reg, n_cells=n_cells, knockout_value=knockout_value, seed=seed)

    return {
        "grn": g,
        "control": control,
        "control_cells": control_cells,
        "targets": targets,
        "train_genes": [int(x) for x in train_genes],
        "test_genes": [int(x) for x in test_genes],
        "knockout_value": knockout_value,
        "simulator": "sergio",
    }


def summarize(ds):
    ctrl = ds["control"]
    moved = []
    for k, tgt in ds["targets"].items():
        moved.append(int(np.sum(np.abs(tgt - ctrl) > 0.05)))
    moved = np.array(moved)
    print(f"simulator={ds.get('simulator', 'toy')}  genes={ds['grn']['n_genes']}  "
          f"perturbations={len(ds['targets'])} "
          f"(train={len(ds['train_genes'])}, test={len(ds['test_genes'])})")
    print(f"per-knockout genes moved >0.05: "
          f"min={moved.min()} median={int(np.median(moved))} max={moved.max()}")
    if "control_cells" in ds:
        cell_std = ds["control_cells"].std(axis=0).mean()
        print(f"single-cell noise: avg per-gene std across "
              f"{ds['control_cells'].shape[0]} simulated cells = {cell_std:.4f}")
