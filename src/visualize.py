"""
visualize.py — the figures that make the project legible (and portfolio-worthy).

  ripple.png         : a held-out knockout's effect spreading through the gene
                       network and settling, step by step. This is the "watchable"
                       payoff — you can literally see the perturbation propagate.
  generalization.png : predicted vs true change on held-out perturbations, model
                       vs the mean-shift baseline.
  baselines.png      : headline metric (Pearson r) for the model vs dumb baselines.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

INK = "#14342b"
ACCENT = "#2d6a4f"


def _graph_layout(structure, seed=3):
    G = nx.from_numpy_array(structure.T, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=seed, k=1.8 / np.sqrt(structure.shape[0]),
                           iterations=200)
    return G, pos


def ripple(model, ds, gene, path, steps=25, frames=6):
    control = ds["control"]
    kv = ds["knockout_value"]
    structure = ds["grn"]["A"]
    _, traj = model(_t(control), int(gene), kv, steps=steps, return_traj=True)
    # traj: (steps+1, n) expression over time; show change from control
    dtraj = traj - control[None, :]

    G, pos = _graph_layout(structure)
    idx = np.linspace(0, len(dtraj) - 1, frames).astype(int)
    # scale color by the RIPPLE, not by the clamp itself: the knocked-out
    # gene is forced to `kv` every step, so its own |change| is trivially
    # ~constant and would otherwise wash out every genuinely propagated
    # effect on the color scale.
    ripple_only = np.delete(dtraj, int(gene), axis=1)
    vmax = max(0.05, np.abs(ripple_only).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = plt.subplots(1, frames, figsize=(3.0 * frames, 3.4))
    xs = [pos[i][0] for i in range(structure.shape[0])]
    ys = [pos[i][1] for i in range(structure.shape[0])]
    for ax, t in zip(axes, idx):
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#cfd8d4",
                               width=0.6, arrows=False, alpha=0.6)
        sc = ax.scatter(xs, ys, c=dtraj[t], cmap="RdBu_r", norm=norm,
                        s=90, edgecolors="none", zorder=3)
        # highlight the knocked-out gene
        ax.scatter([pos[int(gene)][0]], [pos[int(gene)][1]], s=190,
                   facecolors="none", edgecolors=INK, linewidths=2.2, zorder=4)
        ax.set_title(f"step {t}", fontsize=11, color=INK)
        ax.axis("off")
    fig.suptitle(f"Knockout of gene {int(gene)} (held-out) rippling through the "
                 f"network  ·  ringed node = clamped, red = up / blue = down",
                 fontsize=12, color=INK, y=1.02)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("change from control", fontsize=9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def generalization(per_pert, mean_shift, path):
    dp = np.concatenate([v["delta_pred"] for v in per_pert.values()])
    dt = np.concatenate([v["delta_true"] for v in per_pert.values()])
    ms = np.concatenate([mean_shift for _ in per_pert])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, pred, title in [(axes[0], dp, "GraphNCA"),
                            (axes[1], ms, "mean-shift baseline")]:
        lim = max(np.abs(dt).max(), np.abs(pred).max()) * 1.05
        ax.axhline(0, color="#ddd", lw=0.8); ax.axvline(0, color="#ddd", lw=0.8)
        ax.plot([-lim, lim], [-lim, lim], color="#bbb", lw=1, ls="--")
        ax.scatter(dt, pred, s=10, alpha=0.5, color=ACCENT, edgecolors="none")
        r = np.corrcoef(dt, pred)[0, 1] if pred.std() > 1e-8 else 0.0
        ax.set_title(f"{title}   (r = {r:.2f})", color=INK)
        ax.set_xlabel("true change"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    axes[0].set_ylabel("predicted change")
    fig.suptitle("Held-out perturbations: predicted vs true change per gene",
                 color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def single_cell_variability(ds, path, n_show=10, seed=7):
    """Show what the sergio simulator adds over the toy ODE: every simulated
    cell has its own noisy trajectory, so the control 'steady state' is
    really a population with spread, not one number. We show a handful of
    genes' single-cell distributions with the pseudobulk mean (the model's
    actual training target) marked alongside."""
    cells = ds["control_cells"]                # (n_cells, n_genes)
    control = ds["control"]
    n_cells, n_genes = cells.shape
    rng = np.random.default_rng(seed)

    order = np.argsort(-cells.std(axis=0))     # most variable genes first
    show = order[:n_show]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * n_show), 4.2))
    jitter = (rng.random((n_cells, len(show))) - 0.5) * 0.6
    for col, g in enumerate(show):
        ax.scatter(np.full(n_cells, col) + jitter[:, col], cells[:, g],
                   s=6, alpha=0.35, color=ACCENT, edgecolors="none")
        ax.scatter([col], [control[g]], marker="_", s=400, color=INK,
                   linewidths=2, zorder=5)
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels([f"gene {int(g)}" for g in show], rotation=45, ha="right")
    ax.set_ylabel("expression (normalized)")
    ax.set_title(f"Single-cell spread across {n_cells} simulated cells "
                 f"(dash = pseudobulk mean, the model's training target)",
                 color=INK, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def baselines(summary, path):
    names = ["graph-nca", "mean-shift", "no-change"]
    rs = [summary[n]["pearson_r"] for n in names]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(names, rs, color=[ACCENT, "#9cc7b0", "#d5ded9"])
    for b, r in zip(bars, rs):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}",
                ha="center", color=INK, fontsize=10)
    ax.set_ylabel("Pearson r (held-out)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Did it beat the dumb baselines?", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _t(a):
    import torch
    return torch.tensor(a)
