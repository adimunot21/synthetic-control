# synthetic-control

A simulation-only replication study of arXiv [2610.31275](https://arxiv.org/abs/2610.31275), "Learning Robot Behavior Priors from Non-Robotic Synthetic Data" (Odrowaz-Sypniewski, Amir, Bayrooti, Shankar, Prorok; October 2026 preprint, not peer reviewed).

The paper pretrains a transformer on procedurally generated data (random MDPs, bandits and linear dynamical systems) and reports that it predicts robot actions on real datasets better than chance. The paper does not compare against simple baselines. This repo reimplements the generators, builds a five-baseline suite (majority class, kNN, logistic regression, gradient-boosted trees, small MLP) and pretrains a model at laptop scale, about 4.3 hours of training against roughly 30 hours in the paper, to see whether the effect holds.

## Findings

- At this scale the model approaches the baseline suite but does not exceed it, and the gap levels off with more training.
- When autocorrelation in the data is varied, the model's accuracy curve follows a kNN baseline's closely (Pearson r = 0.991).
- Three further probes agree: the model's predictions and mistakes overlap with kNN far more than chance, it shows no extra robustness on the one system that needs extrapolation, and it is invariant to relabelling the actions.
- One positive result: a measurable sample-efficiency advantage over kNN at small context sizes, which disappears as context grows.

All results come from three simulated systems (pendulum, cartpole, Dubins car). Section 6 of `docs/report.md` lists what the study does not show.

Status: phases 0 to 2 complete.

## Documents

- `docs/report.md`: full write-up with figures and ablations.
- `docs/MODEL_CARD.md`: model and dataset card.
- `docs/00_critique_and_plan.md`: critique of the paper, related work, plan and risks.
- `docs/data-contract-tabpfn.md`: inputs, outputs and required patches for `tabpfn==0.1.11`.
- `docs/reimplementation-assumptions.md`: every detail filled in beyond what the paper describes.

## Setup

    conda create -n synthetic-control python=3.11 -y
    conda activate synthetic-control
    pip install -r requirements.txt
    pip install -e .
    python scripts/patch_tabpfn_torch_compat.py    # one-time compatibility patch
    pytest tests/ -v

Reproducing the main results needs a trained checkpoint (`scripts/train_control_prior_smoke.py`, `scripts/extend_training.py`):

    python scripts/run_baseline_suite_phase1.py
    python scripts/run_rq1_decision_gate.py
    python scripts/rq2_autocorrelation_dose_response.py
    python scripts/rq3_mechanism_probes.py
    python scripts/make_report_figures.py

## Code layout

- `src/synctrl/sims/`: pendulum, cartpole and Dubins car with randomised physical parameters and a closed-form oracle accuracy ceiling.
- `src/synctrl/generators/`: reimplementations of the paper's MDP, bandit and LDS generators, and a generic tabular prior.
- `src/synctrl/baselines/`: the five baselines, fitted fresh for each task.
- `src/synctrl/model.py`, `pretrain.py`: the model and its pretraining loop.
- `src/synctrl/eval/`: accuracy and NLL metrics, with paired bootstrap confidence intervals.
- `tests/`: 77 tests covering simulators, generators, baselines, model and evaluation.
