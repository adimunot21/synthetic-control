# Model & Dataset Card

Covers the trained control-prior model and the synthetic generators used to pretrain it. Companion to [docs/report.md](report.md), which has the full results and analysis — this card is the short reference.

## Model

**What it is.** A `tabpfn.transformer.TransformerModel` instance (64-dim embeddings, 4 attention heads, 128-dim feedforward, 4 layers) trained from scratch via in-context next-action prediction on synthetic data only. No real-world or real-robot data was used in training or evaluation anywhere in this project.

**Training data.** A uniform mixture of three procedurally generated task distributions, reimplemented from arXiv 2610.31275 Section IV-B: tabular MDPs, contextual bandits, and linear dynamical systems (see [src/synctrl/generators/](../src/synctrl/generators/) and [docs/reimplementation-assumptions.md](reimplementation-assumptions.md) for exact specifications and every filled-in detail not given by the source paper). 16,800 training steps, batch size 16, sequence length 200, ~270,000 total sampled tasks, ~4.3 hours on a single consumer GPU (NVIDIA GTX 1650).

**Intended use.** Research only — characterizing whether and how synthetic-control-prior pretraining transfers to simulated control tasks, and what mechanism explains any transfer observed. **Not intended for, and not evaluated for, any deployment use, real robot control, or any application beyond this study.**

**Known limitations (see [docs/report.md](report.md) §5-6 for full detail):**
- Trained at roughly 1/1000th the scale of the source paper's own setup (steps, dataset count, and training time combined); results do not generalize to claims about that paper's actual configuration.
- Evaluated on exactly three simulated systems (pendulum, cartpole, Dubins car) — a narrow slice of control problems, and all strictly simulated.
- Behavior is best explained by local (kNN-like) structure rather than learned long-horizon control reasoning (see report §3, §7); it should not be presented or used as evidence of general control competence.
- Not evaluated in closed-loop control at all — only open-loop, single-step-ahead in-context action prediction.

**Performance summary.** Approaches but does not exceed a simple baseline suite (majority class, kNN, logistic regression, gradient-boosted trees, small MLP) on held-out accuracy across all three evaluation systems and both split protocols tested. Full numbers in [results/phase1_rq1_decision_gate.json](../results/phase1_rq1_decision_gate.json) and [docs/report.md](report.md).

## Dataset (synthetic generators)

**Source/provenance.** Reimplemented from the written description in arXiv 2610.31275 (not peer reviewed as of this writing); no official code or data release is available at time of writing. Generated procedurally at runtime — no generator output is stored as a static dataset; every training batch is freshly sampled.

| Generator | What it models | Key parameters (see generator CONFIG dicts for exact ranges) |
|---|---|---|
| MDP | Long-horizon value-driven decision-making | state dim, # states, # actions, value-iteration discount/horizon |
| Contextual Bandit | Single-step state-action association, no temporal structure | state dim, action feature dim, reward bilinear rank |
| LDS | Smooth, highly autocorrelated continuous dynamics | state dim, decay rate (autocorrelation strength), process noise |

**Evaluation data.** Three known-expert simulated systems ([src/synctrl/sims/](../src/synctrl/sims/)), also generated procedurally with randomized physical parameters, never overlapping with the synthetic pretraining generators' task family:

| System | Action space | Expert policy |
|---|---|---|
| Pendulum | 3 discrete torques | Energy-shaping swing-up + PD balance |
| Cartpole | 2 discrete forces | PD bang-bang |
| Dubins car | 3 discrete steering commands | Proportional heading pursuit |

**License note.** This project's own code is original work in this repository. It depends on `tabpfn==0.1.11`, confirmed MIT-licensed per its PyPI trove classifier (see [docs/data-contract-tabpfn.md](data-contract-tabpfn.md) for a documented licensing inconsistency in that package's own bundled text, and the one-line torch-compatibility patch required to run it on modern PyTorch).

## Ethical / scope considerations

This is a simulation-only research study investigating a specific published claim. No personal data, no real robots, and no deployment of any kind is involved. The headline finding is a measured negative-to-mixed result at the tested scale — reported plainly rather than selectively, per the project's own stated commitment to report negative results as such.
