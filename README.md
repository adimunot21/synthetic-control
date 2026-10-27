# synthetic-control

Independent, simulation-only investigation of arXiv [2610.31275](https://arxiv.org/abs/2610.31275), "Learning Robot Behavior Priors from Non-Robotic Synthetic Data" (Odrowaz-Sypniewski, Amir, Bayrooti, Shankar, Prorok; Cambridge; 22 Oct 2026 preprint, not peer reviewed).

Status: **Phase 0 (setup) complete.** See [docs/00_critique_and_plan.md](docs/00_critique_and_plan.md) for the paper critique, verified related work, and the full phase/milestone/risk-register plan.

## Scope

All simulation. No real hardware, no real-robot datasets. Every result is scoped to the simulated systems actually tested here — see the critique doc for the full set of scope constraints.

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

## Layout

- `src/synctrl/sims/` — known-expert simulated systems (pendulum, cartpole, Dubins car) with randomized physical parameters and a closed-form oracle accuracy ceiling under policy noise. See [docs/00_critique_and_plan.md](docs/00_critique_and_plan.md) §5 Phase 0.
- `docs/00_critique_and_plan.md` — paper critique, related-work verification, engineering feasibility checks, phase plan, risk register.
- `docs/data-contract-tabpfn.md` — inspected I/O contract, required patches, and known upstream quirks for `tabpfn==0.1.11`, which this project reuses for its generic tabular-prior baseline.
- `scripts/patch_tabpfn_torch_compat.py` — one-time, idempotent compatibility patch (tabpfn 0.1.11 predates a torch internals change).
- `scripts/smoke_test_phase0.py` — end-to-end shape/contract smoke test: simulator rollout → context/query split → tiny TabPFN forward pass.
- `tests/` — unit tests for the simulators (rollout shapes, oracle-ceiling statistics, parameter ranges, control-objective sanity checks).
