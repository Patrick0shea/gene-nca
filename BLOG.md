# Teaching a cell to heal itself, in software

*Or: what happens when you point a Neural Cellular Automaton at a gene-regulatory network instead of a picture of a lizard.*

In 2020, a paper called [*Growing Neural Cellular Automata*](https://distill.pub/2020/growing-ca/) showed something a little eerie: a grid of cells, each running the exact same tiny learned rule, each only able to see its immediate neighbors, could grow into a picture of a lizard emoji — and if you deleted half the picture mid-simulation, it would *regrow*. No cell knew the global picture. No cell had special instructions. Just one shared local rule, applied over and over, that happened to encode "keep converging toward the target pattern no matter what state you're currently in."

That "keep converging back to a stable pattern after damage" property is also, more or less, a description of what a living cell does after you knock out one of its genes. Its gene-regulatory network doesn't collapse — it re-settles into a new equilibrium, propagating the loss through everything downstream of that gene. So: what if the "grid" in a neural cellular automaton wasn't a 2D image, but a gene-regulatory graph? What if "damage" wasn't an eraser, but a CRISPR knockout — pinning one gene's expression to zero and asking the automaton to relax around it?

That's `gene-nca`.

## The setup

The model — I'm calling it **GraphNCA** — gets a gene-regulatory graph (which genes regulate which, nothing else: no edge signs, no weights, no equations) and a starting expression state. Every step, every gene looks at its regulators and its targets, combines that with its own state through a small shared MLP, and nudges itself. Run that for 25 steps and it settles. To predict a knockout, you clamp the perturbed gene to its silenced value every single step and let everything else relax around the hole.

Crucially, the model is trained on knockouts of *one* set of genes and evaluated on knockouts of genes it has **never seen perturbed**. It only gets to see the graph structure and infer, from other genes' training behavior, how a completely novel clamp should propagate. That's the whole test. A model that just memorizes "knock out gene 12, gene 12 was always X" learns nothing useful — it has to have actually internalized *how effects spread through this topology*.

Against two deliberately dumb baselines (predict no change; predict the *average* training response for every test gene), GraphNCA wins clearly — more on the actual numbers below.

## Sandbox first, on purpose

This is explicitly a synthetic sandbox, and I mean explicitly — the very first line of the README says so. The gene network is one we generate ourselves, so we know the *true* answer to every possible knockout and can grade the model exactly, before ever touching real, messy, partially-wrong biological data. If the model can't learn causal propagation on data where we control the ground truth, it has no business anywhere near a real Perturb-seq dataset.

## Making the sandbox honest: swapping in SERGIO-style kinetics

The first version's ground-truth generator was about as simple as it gets: one global sigmoid squashing a linear combination of a gene's regulators, iterated to a fixed point. It worked, but it was too clean — one deterministic number per gene, no such thing as a "cell," no noise.

The natural next step, and the first item on the roadmap, was to swap that out for something closer to [**SERGIO**](https://github.com/PayamDiba/SERGIO) — a well-regarded single-cell expression simulator built specifically around known gene-regulatory networks. Real single-cell realism: Hill-function regulatory kinetics (the actual saturating, switch-like curves real transcription factors produce, not a flat sigmoid over a sum), simulated stochastically *per cell* via the Chemical Langevin Equation, so that "the control state" is a whole noisy population of cells, not one number.

Small snag: SERGIO isn't on PyPI. It's GitHub-only, and the sandboxed environment I was working in can't fetch arbitrary GitHub repos. I could have quietly downgraded the ambition here and called it a day. Instead I went back to the actual paper — Dibaeinia & Sinha, 2020, *Cell Systems* — and reimplemented the algorithm it describes: master regulators with a fixed basal production rate, target genes whose production rate is a sum of Hill-function contributions from their regulators (an increasing curve for an activator, a decreasing one for a repressor), the whole population integrated cell-by-cell with real production/decay noise, plus an optional dropout and library-size layer for realism. A knockout still means clamping one gene's concentration at every integration step — same idea as before, just now the thing being clamped is a genuinely stochastic multi-cell simulation instead of a single deterministic trajectory.

I want to be upfront about this rather than bury it: `sergio_sim.py` is *my* reimplementation of SERGIO's published model, not the actual `PayamDiba/SERGIO` codebase, and it hasn't been numerically cross-checked against the reference implementation. It's a faithful-to-the-paper reconstruction, not a vendored dependency. The README says the same thing, in the same words, right where the module is introduced — no reason for a blog post to be more diplomatic about it than the code comments are.

## The figure that came out blank

Once the new simulator was wired in and the model retrained, I generated the "ripple" figure — the one meant to show a knockout's effect spreading outward through the network, frame by frame. It came back almost entirely white. Just the ringed clamped node, and otherwise nothing.

The bug, once I went looking, was almost embarrassingly simple. A knockout works by forcing the perturbed gene's expression to 0 for the whole simulation. That gene's own "change from control" is therefore always huge and — this is the important part — *completely uninteresting*, because it's not a discovered effect, it's the intervention itself. The plotting code was computing its color scale from the maximum change across *every* gene, clamped one included. So the color scale stretched out to cover that one trivial, guaranteed-large data point, and the genuinely interesting stuff — how much everything *downstream* actually moved — got squeezed into a sliver near zero and rendered as basically nothing.

There was a second copy of the same mistake: the code that picks *which* held-out knockout is "most interesting" to animate was ranking candidates by total change including that same trivial term — so it kept picking whichever gene happened to have the largest resting expression value, not the one whose knockout actually rippled furthest.

Excluding the clamped gene from both the color-scale calculation and the interestingness ranking fixed both problems at once. The animation at the top of this post — and the ripple figure in the repo — is the result: color visibly spreading outward from the clamped node as the automaton settles.

It's a small bug, but I think it's a good illustration of a general trap in this kind of visualization work: the "before/after" delta on the thing you *intervened on* is not a result, it's a definition, and it will silently dominate any shared scale you compute alongside actual results if you let it.

## Where it landed

With the harder, noisier SERGIO-style ground truth in place, the headline numbers came down — from a suspiciously clean r ≈ 0.91 on the old toy ODE to r ≈ 0.70–0.77 across seeds on the new simulator. That's the point, not a regression: the new data dilutes each knockout's effect across a target's other active regulators and adds real stochastic noise on top, so the task got honestly harder. GraphNCA still separates clearly from both dumb baselines (mean-shift sits around r ≈ 0.2, no-change is r = 0 by construction), which is the thing that actually matters — the model is still recovering real causal structure, just on data that no longer hands it an easy win.

| method | Pearson r (held-out) | sign accuracy on true DEGs |
|---|---|---|
| **GraphNCA** | **0.77** | **0.65** |
| mean-shift baseline | 0.21 | 0.54 |
| no-change baseline | 0.00 | 0.00 |

## What's next

The roadmap in the repo is honest about how far this is from a paper: a real Perturb-seq slice benchmarked against an actual published baseline (Arc Institute's STATE model), scored with standard tooling (Cell-Eval / PerturBench) so the numbers mean something to anyone outside this repo; robustness to a graph that's noisy or incomplete, the way real regulatory networks always are; letting the model consume the single-cell variability the new simulator produces instead of only training against the pseudobulk mean; and eventually relaxing the assumption that the graph is given at all.

None of that is done. What *is* done is a working, honestly-graded proof that the core idea — a shared local update rule on a gene graph can learn to propagate a novel perturbation it's never seen, purely from topology and a handful of training examples — holds up even once the sandbox stops being easy.

Code: [`gene-nca`](.)
