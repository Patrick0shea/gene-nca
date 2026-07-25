# gene-nca

**A cell modelled as a neural cellular automaton over its gene-regulatory network — used to predict how the cell responds to genetic perturbations.**

Most models that predict perturbation responses treat the cell as a black box. This one borrows the idea behind *Growing Neural Cellular Automata* (Mordvintsev et al., 2020): a single shared local rule, applied to every unit over and over, until the system settles — the same process that lets those models grow a pattern and *heal it back* after damage. Here the "grid" is replaced by the gene-regulatory graph, and "healing back to a stable state after a disturbance" is exactly what a cell does after you knock out a gene.

To predict a knockout, we pin the perturbed gene's expression to its silenced value and let the automaton relax. The settled state is the prediction.

> This repo is the **sandbox-first** stage: a synthetic gene network where we control the wiring, so we know the true answer for every perturbation and can prove the core idea works before touching messy real data.

## The result

Trained on knockouts of one set of genes, the model predicts knockouts of genes it **never saw perturbed**, purely by propagating a novel clamp through the learned network dynamics. It is compared against two deliberately dumb baselines — because a model that only ties a dumb baseline is not a finding.

| method | Pearson r (held-out) | sign accuracy on true DEGs | MSE |
|---|---|---|---|
| **GraphNCA** | **0.913** | **0.756** | **0.00054** |
| mean-shift baseline | −0.123 | 0.494 | 0.00676 |
| no-change baseline | 0.000 | 0.000 | 0.00658 |

Stable across seeds (r ≈ 0.91–0.92, sign accuracy ≈ 0.76–0.82; baselines stay at chance or worse). `no-change` predicts the control for everything; `mean-shift` predicts the *average* training response for every test perturbation — it captures the typical ripple but cannot specialise to the specific gene knocked out, which is the whole game.

### The perturbation rippling through the network and settling
![ripple](figures/ripple.png)

*A held-out knockout (ringed node) spreading hop-by-hop through the gene graph. Red = up-regulated, blue = down-regulated, relative to the unperturbed control. This is a genuine readout of the predicted mechanism, not a cartoon.*

### Held-out generalization
![generalization](figures/generalization.png)

### Beating the baselines
![baselines](figures/baselines.png)

## Run it

```bash
pip install -r requirements.txt
python run.py
```

Runs in ~1–2 minutes on CPU. Figures land in `figures/`. Flags: `--genes`, `--seed`, `--steps`, `--epochs`.

## How it works

```
src/
  grn.py        synthetic gene-regulatory network + ground-truth steady-state
                simulator (a cell state = the network's attractor; a knockout =
                clamp a gene and re-settle)
  data.py       builds the supervised dataset and the train/test split over
                PERTURBED GENES (held-out generalization test)
  model.py      GraphNCA: message passing over the gene graph + a shared residual
                update rule, iterated to a fixed point, with the perturbed gene
                clamped every step
  train.py      backprop-through-time over the unrolled settling steps, with a
                weighted MSE that emphasises genes that actually move
  evaluate.py   held-out metrics vs the dumb baselines
  visualize.py  the ripple animation frames + generalization + baseline figures
run.py          generate -> train -> evaluate -> figures
```

The model is given **only the binary graph structure** (which genes regulate which). It is *not* given the true edge signs, weights, basal drives, or the generator's equations — it has to learn the rule from the training perturbations alone.

## Honest caveats

- **This is a proof-of-concept, not proof it beats real methods.** The synthetic generator and the model share an inductive bias (both settle to an attractor on a graph), so strong sandbox performance shows the model *can* learn and generalise perturbation dynamics — it does **not** show it wins on real data. That's the next stage, and the point of keeping the sandbox as a permanent debugging rig.
- **Graph quality matters.** Here the structure is perfect. On real data the regulatory graph is noisy and partial; robustness to bad edges is itself an experiment worth running.
- **One settled state ≈ pseudobulk.** The model predicts a single resting state, which maps cleanly onto pseudobulk-level metrics (the ones that mattered most in the 2025 Virtual Cell Challenge) but is not yet a model of cell-to-cell variability.

## Roadmap

1. **Swap the toy simulator for [SERGIO](https://github.com/PayamDiba/SERGIO)** — richer, more realistic synthetic single-cell data from a known GRN, so the sandbox stays honest but gets harder.
2. **Real Perturb-seq slice** — reproduce a known baseline (e.g. Arc Institute's open **STATE** model) before claiming anything, and score with **Cell-Eval / PerturBench** so numbers are comparable to the field.
3. **Virtual Cell Challenge** — plug into the held-out-cell-type task; the natural contribution is a rigorous "where does this lens win and where does it lose" analysis, plus the interpretable ripple as a mechanism readout.
4. **Learn the graph** — currently the structure is given; relax that and let edges be inferred or refined.

## Reference

- Mordvintsev, Randazzo, Niklasson, Levin (2020). *Growing Neural Cellular Automata.* Distill.
- Arc Institute — Virtual Cell Challenge (task, data, metrics, STATE baseline).

MIT licensed.
