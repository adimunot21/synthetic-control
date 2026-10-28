# synthetic-control

Independent, simulation-only investigation of arXiv [2610.31275](https://arxiv.org/abs/2610.31275), "Learning Robot Behavior Priors from Non-Robotic Synthetic Data" (Odrowaz-Sypniewski, Amir, Bayrooti, Shankar, Prorok; Cambridge; 22 Oct 2026 preprint, not peer reviewed).

Status: **Phases 0-2 complete.** Full report below the headline finding.

## Headline finding

The paper claims a transformer pretrained only on procedurally generated, non-robotic synthetic data (random MDPs, bandits, linear dynamical systems) predicts robot actions above chance on real datasets it never saw — but never tests against a single simple baseline. We reimplemented its generators, built a five-baseline suite (majority class, kNN, logistic regression, gradient-boosted trees, small MLP), and pretrained a model at laptop scale (~4.3h, ~3 orders of magnitude below the paper's own ~30h/15M-dataset budget) to test whether the effect survives.

**At this scale, it doesn't clearly survive, and the mechanism looks different from what's claimed.** The model approaches but never exceeds the baseline suite, and the gap plateaus rather than closing with more training. A direct manipulation of autocorrelation — the one property the paper's own ablation flags as most important — shows the model's accuracy curve tracking a plain kNN baseline's almost exactly (Pearson r = 0.991). Three further mechanism probes point the same direction: the model agrees with kNN far more than chance (including on shared mistakes), shows no special robustness on the one evaluation system that requires real extrapolation, and is behaviorally invariant to action-label identity (ruling out one specific failure mode, a spurious numeric-label-order dependency). The one genuine, positive finding is narrower than "it learned control": a measurable sample-efficiency edge over kNN at small context sizes that vanishes as more context becomes available.

**Full report, figures, ablations, and an explicit "what this does not show" section: [docs/report.md](docs/report.md).** Model/dataset card: [docs/MODEL_CARD.md](docs/MODEL_CARD.md). Paper critique and project plan: [docs/00_critique_and_plan.md](docs/00_critique_and_plan.md).

## Scope

All simulation. No real hardware, no real-robot datasets. Every result is scoped to the three simulated systems actually tested here (pendulum, cartpole, Dubins car) — see [docs/report.md](docs/report.md) §6 for the full "what this does not show."

## Setup

```bash
conda create -n synthetic-control python=3.11 -y
conda activate synthetic-control
pip install -r requirements.txt
pip install -e .
python scripts/patch_tabpfn_torch_compat.py  # required one-time compat fix, see docs/data-contract-tabpfn.md
```

Verify the install:

```bash
pytest tests/ -v
python scripts/smoke_test_phase0.py
```

Reproduce the main results (requires a trained checkpoint — see `scripts/train_control_prior_smoke.py` / `scripts/extend_training.py`):

```bash
python scripts/run_baseline_suite_phase1.py        # baselines only, no pretraining
python scripts/run_rq1_decision_gate.py            # model vs. baselines, paired comparison
python scripts/rq2_autocorrelation_dose_response.py
python scripts/rq3_mechanism_probes.py
python scripts/make_report_figures.py              # regenerates results/figures/*.png from saved results
```

## Layout

- `src/synctrl/sims/` — known-expert simulated systems (pendulum, cartpole, Dubins car) with randomized physical parameters and a closed-form oracle accuracy ceiling under policy noise.
- `src/synctrl/generators/` — reimplementations of the paper's MDP/Bandit/LDS synthetic control-prior generators, plus a generic tabular-prior baseline generator.
- `src/synctrl/baselines/` — the five-baseline suite (majority, kNN, logistic regression, GBT, small MLP), fit fresh per task.
- `src/synctrl/model.py`, `src/synctrl/pretrain.py` — the control-prior model and its pretraining loop.
- `src/synctrl/eval/` — metrics (accuracy, NLL) and the evaluation harness with paired bootstrap confidence intervals.
- `docs/report.md` — the full workshop-style writeup: method, results, ablations, limitations, "what this does not show."
- `docs/MODEL_CARD.md` — model and dataset card.
- `docs/00_critique_and_plan.md` — paper critique, related-work verification, engineering feasibility checks, phase plan, risk register, and a results summary (§7).
- `docs/data-contract-tabpfn.md` — inspected I/O contract, required patches, and known upstream quirks for `tabpfn==0.1.11`.
- `docs/reimplementation-assumptions.md` — every detail filled in beyond the source paper's written description, logged rather than silently assumed.
- `results/` — saved JSON results and figures, versioned alongside the code that produced them.
- `tests/` — unit and statistical-property tests (77 passing) for the simulators, generators, baselines, model, and evaluation harness.
