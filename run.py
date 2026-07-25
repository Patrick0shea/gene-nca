"""
run.py — end to end.

    python run.py

Generates a synthetic GRN, trains the GraphNCA on training-gene knockouts,
evaluates on held-out knockouts against dumb baselines, and writes the figures.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from src.data import build_dataset, summarize
from src.model import GraphNCA
from src.train import train
from src.evaluate import evaluate, print_table
from src import visualize as viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=250)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("1) building synthetic GRN + ground-truth perturbations")
    ds = build_dataset(n_genes=args.genes, seed=args.seed)
    summarize(ds)

    print("\n2) training GraphNCA on held-in knockouts")
    model = GraphNCA(ds["grn"]["A"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   model params: {n_params:,}")
    train(model, ds, steps=args.steps, epochs=args.epochs, seed=args.seed)

    print("\n3) evaluating on HELD-OUT knockouts vs baselines")
    summary, per_pert, mean_shift = evaluate(model, ds, steps=args.steps)
    print_table(summary)

    print("\n4) writing figures/")
    # choose a held-out perturbation with a big, interesting ripple to animate
    impact = {k: np.abs(v["delta_true"]).sum() for k, v in per_pert.items()}
    show_gene = max(impact, key=impact.get)
    viz.ripple(model, ds, show_gene, "figures/ripple.png", steps=args.steps)
    viz.generalization(per_pert, mean_shift, "figures/generalization.png")
    viz.baselines(summary, "figures/baselines.png")
    print("   figures/ripple.png  figures/generalization.png  figures/baselines.png")

    print("\ndone.")
    return summary


if __name__ == "__main__":
    main()
