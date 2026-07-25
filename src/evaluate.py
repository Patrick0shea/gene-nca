"""
evaluate.py — grade the model on HELD-OUT perturbations against dumb baselines.

The point of the sandbox (straight from the first conversation): a giant model
that only ties a deliberately dumb baseline is not interesting. So we always
report the baselines alongside the model.

Metrics, per perturbation, averaged over the held-out set:
  - Pearson r between predicted and true change (delta = perturbed - control).
    This is the discrimination-style signal: did you predict the right RIPPLE?
  - Sign accuracy on the true DEGs (genes that actually moved): did you get the
    up/down direction right where it matters?
  - MSE over all genes.

Baselines:
  - "no-change"   : predict control everywhere (delta = 0).
  - "mean-shift"  : predict the average training-perturbation response for EVERY
                    test perturbation (captures the typical response but cannot
                    specialise to the specific gene knocked out).
"""
from __future__ import annotations
import numpy as np


def _pearson(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _metrics_for(delta_pred, delta_true, deg_thresh=0.05):
    r = _pearson(delta_pred, delta_true)
    mse = float(np.mean((delta_pred - delta_true) ** 2))
    deg = np.abs(delta_true) > deg_thresh
    if deg.sum() == 0:
        sign_acc = np.nan
    else:
        sign_acc = float(np.mean(
            np.sign(delta_pred[deg]) == np.sign(delta_true[deg])))
    return r, sign_acc, mse


def evaluate(model, ds, steps=25):
    control = ds["control"]
    kv = ds["knockout_value"]
    test_genes = ds["test_genes"]
    train_genes = ds["train_genes"]
    targets = ds["targets"]

    mean_shift = np.mean(
        [targets[k] - control for k in train_genes], axis=0)

    rows = {"graph-nca": [], "mean-shift": [], "no-change": []}
    per_pert = {}
    for k in test_genes:
        dt = targets[k] - control
        pred = model.predict(control, k, kv, steps=steps)
        dp = pred - control
        rows["graph-nca"].append(_metrics_for(dp, dt))
        rows["mean-shift"].append(_metrics_for(mean_shift, dt))
        rows["no-change"].append(_metrics_for(np.zeros_like(dt), dt))
        per_pert[k] = {"delta_true": dt, "delta_pred": dp}

    summary = {}
    for name, vals in rows.items():
        arr = np.array([[v[0], v[1], v[2]] for v in vals], dtype=float)
        summary[name] = {
            "pearson_r": np.nanmean(arr[:, 0]),
            "sign_acc": np.nanmean(arr[:, 1]),
            "mse": np.nanmean(arr[:, 2]),
        }
    return summary, per_pert, mean_shift


def print_table(summary):
    print(f"\n  {'method':<12}{'Pearson r':>12}{'sign acc':>12}{'MSE':>12}")
    print("  " + "-" * 46)
    for name in ["graph-nca", "mean-shift", "no-change"]:
        s = summary[name]
        sign = "n/a" if np.isnan(s["sign_acc"]) else f"{s['sign_acc']:.3f}"
        print(f"  {name:<12}{s['pearson_r']:>12.3f}{sign:>12}{s['mse']:>12.5f}")
