"""RQ3 (docs/00_critique_and_plan.md): what is the model actually doing?

Four probes against the trained control-prior model on held-out real
systems (pendulum/cartpole/dubins), none of which try to answer the
question from accuracy numbers alone:

(a) kNN-agreement rate: how often the model's prediction literally matches
    kNN's, beyond what their individual accuracies would predict under
    independence -- a mechanistic signal, not just a performance comparison.
(b) Context-size scaling: reproduces the paper's Section V-E curve on our
    own systems, model vs. kNN, to compare how each benefits from more
    context.
(c) The model's own contiguous-vs-random split gap, read directly from
    results/phase1_rq1_decision_gate.json (already computed, no new runs) --
    does the model show the same large Dubins leakage-gap pattern the
    baselines did, or does it generalize differently?
(d) Label-permutation consistency: TabPFN's y_encoder embeds action labels
    via a linear projection of their *numeric* value (nn.Linear(1, ninp)),
    not as unordered categorical symbols -- so relabeling actions within a
    task via a random permutation (applied consistently to context and
    query) is a genuine probe of whether the model has learned a spurious
    dependence on label ordering, not a free no-op.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from synctrl.baselines.knn import KNNBaseline
from synctrl.eval.metrics import paired_difference_ci, task_accuracy
from synctrl.model import ControlPriorModel, build_model
from synctrl.sims.cartpole import CartPole
from synctrl.sims.dubins import DubinsCar
from synctrl.sims.pendulum import Pendulum
from synctrl.tasks import Task, sample_tasks, task_from_arrays

SYSTEMS = {"pendulum": Pendulum(), "cartpole": CartPole(), "dubins": DubinsCar()}
CHECKPOINT = "checkpoints/step_16800.pt"
SEED = 0


def load_model(checkpoint_path: str = CHECKPOINT) -> ControlPriorModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    model = build_model(
        num_features=config["num_features"], n_classes=config["max_actions"],
        ninp=config["ninp"], nhead=config["nhead"], nhid=config["nhid"], nlayers=config["nlayers"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return ControlPriorModel(
        model, num_features=config["num_features"], n_out=config["max_actions"], device=device,
        rng=np.random.default_rng(SEED),
    )


# ---------------------------------------------------------------------------
# (a) kNN-agreement rate
# ---------------------------------------------------------------------------

def knn_agreement_probe(model: ControlPriorModel) -> dict:
    print("\n=== (a) kNN-agreement rate ===")
    knn = KNNBaseline(k=5)
    results = {}
    for name, system in SYSTEMS.items():
        rng = np.random.default_rng(SEED)
        tasks = sample_tasks(system, n_tasks=60, n_steps=200, rng=rng, noise_eps=0.15, split="contiguous")

        agreements, model_accs, knn_accs, expected_independent = [], [], [], []
        for task in tasks:
            model_proba = model.predict_proba(task.context_x, task.context_y, task.query_x, task.action_space_size)
            knn_proba = knn.predict_proba(task.context_x, task.context_y, task.query_x, task.action_space_size)
            model_pred = model_proba.argmax(axis=1)
            knn_pred = knn_proba.argmax(axis=1)

            agreement = float(np.mean(model_pred == knn_pred))
            model_acc = task_accuracy(model_proba, task.query_y)
            knn_acc = task_accuracy(knn_proba, task.query_y)
            n_classes = task.action_space_size
            # If model/kNN errors were independent, expected agreement rate
            # is: both correct (and thus agreeing) + both wrong and
            # (by chance) landing on the same one of the remaining classes.
            expected = model_acc * knn_acc + (1 - model_acc) * (1 - knn_acc) / max(n_classes - 1, 1)

            agreements.append(agreement)
            model_accs.append(model_acc)
            knn_accs.append(knn_acc)
            expected_independent.append(expected)

        observed = float(np.mean(agreements))
        expected_mean = float(np.mean(expected_independent))
        results[name] = {
            "observed_agreement": observed,
            "expected_if_independent": expected_mean,
            "model_accuracy": float(np.mean(model_accs)),
            "knn_accuracy": float(np.mean(knn_accs)),
        }
        print(
            f"{name:<10} observed agreement {observed:.3f} vs expected-if-independent {expected_mean:.3f} "
            f"(model_acc {np.mean(model_accs):.3f}, knn_acc {np.mean(knn_accs):.3f})"
        )
    return results


# ---------------------------------------------------------------------------
# (b) Context-size scaling
# ---------------------------------------------------------------------------

def context_size_probe(model: ControlPriorModel) -> dict:
    print("\n=== (b) Context-size scaling (fixed query_size=20) ===")
    knn = KNNBaseline(k=5)
    context_sizes = [10, 20, 40, 60, 80, 100, 150, 190]
    query_size = 20
    results = {}

    for system_name, system in SYSTEMS.items():
        results[system_name] = {}
        for context_size in context_sizes:
            seq_len = context_size + query_size
            rng = np.random.default_rng(SEED)
            model_accs, knn_accs = [], []
            for _ in range(40):
                rollout = system.rollout(n_steps=seq_len, rng=rng, noise_eps=0.15)
                x_all, y_all = rollout.states[:-1], rollout.actions
                context_x, query_x = x_all[:context_size], x_all[context_size:]
                context_y, query_y = y_all[:context_size], y_all[context_size:]

                model_proba = model.predict_proba(context_x, context_y, query_x, system.action_space_size)
                knn_proba = knn.predict_proba(context_x, context_y, query_x, system.action_space_size)
                model_accs.append(task_accuracy(model_proba, query_y))
                knn_accs.append(task_accuracy(knn_proba, query_y))

            results[system_name][context_size] = {
                "model_accuracy": float(np.mean(model_accs)),
                "knn_accuracy": float(np.mean(knn_accs)),
            }
        row = results[system_name]
        print(f"{system_name}:")
        print("  context_size: " + " ".join(f"{c:>6d}" for c in context_sizes))
        print("  model_acc:    " + " ".join(f"{row[c]['model_accuracy']:>6.3f}" for c in context_sizes))
        print("  knn_acc:      " + " ".join(f"{row[c]['knn_accuracy']:>6.3f}" for c in context_sizes))
    return results


# ---------------------------------------------------------------------------
# (c) Model's own contiguous-vs-random gap -- free, from existing results
# ---------------------------------------------------------------------------

def contiguous_vs_random_gap_from_existing_results() -> dict:
    print("\n=== (c) Model's own contiguous-vs-random split gap (from existing results) ===")
    with open("results/phase1_rq1_decision_gate.json") as f:
        data = json.load(f)

    results = {}
    for system_name, by_split in data.items():
        model_contig = by_split["contiguous"]["control_prior_model"]["aggregate"]["accuracy"]["mean"]
        model_random = by_split["random"]["control_prior_model"]["aggregate"]["accuracy"]["mean"]
        best_baseline_contig = max(
            (v["aggregate"]["accuracy"]["mean"] for k, v in by_split["contiguous"].items() if k != "control_prior_model")
        )
        best_baseline_random = max(
            (v["aggregate"]["accuracy"]["mean"] for k, v in by_split["random"].items() if k != "control_prior_model")
        )
        results[system_name] = {
            "model_gap": model_random - model_contig,
            "best_baseline_gap": best_baseline_random - best_baseline_contig,
        }
        print(
            f"{system_name:<10} model random-contiguous gap: {model_random - model_contig:+.3f}  |  "
            f"best baseline's gap: {best_baseline_random - best_baseline_contig:+.3f}"
        )
    return results


# ---------------------------------------------------------------------------
# (d) Label-permutation consistency
# ---------------------------------------------------------------------------

def label_permutation_probe(model: ControlPriorModel) -> dict:
    print("\n=== (d) Label-permutation consistency ===")
    results = {}
    for name, system in SYSTEMS.items():
        rng = np.random.default_rng(SEED)
        tasks = sample_tasks(system, n_tasks=60, n_steps=200, rng=rng, noise_eps=0.15, split="contiguous")

        original_accs, permuted_accs = [], []
        for task in tasks:
            sigma = rng.permutation(task.action_space_size)

            original_proba = model.predict_proba(task.context_x, task.context_y, task.query_x, task.action_space_size)
            original_accs.append(task_accuracy(original_proba, task.query_y))

            context_y_perm = sigma[task.context_y]
            query_y_perm = sigma[task.query_y]
            permuted_proba = model.predict_proba(task.context_x, context_y_perm, task.query_x, task.action_space_size)
            permuted_accs.append(task_accuracy(permuted_proba, query_y_perm))

        gap = paired_difference_ci(permuted_accs, original_accs, rng)
        results[name] = {
            "original_accuracy": float(np.mean(original_accs)),
            "permuted_accuracy": float(np.mean(permuted_accs)),
            "gap_mean": gap.mean, "gap_ci_lo": gap.ci_lo, "gap_ci_hi": gap.ci_hi,
        }
        verdict = "label-order-invariant" if gap.ci_lo <= 0 <= gap.ci_hi else "DEPENDS on label order"
        print(
            f"{name:<10} original_acc {np.mean(original_accs):.3f}  permuted_acc {np.mean(permuted_accs):.3f}  "
            f"gap {gap.mean:+.3f} [{gap.ci_lo:+.3f}, {gap.ci_hi:+.3f}]  -> {verdict}"
        )
    return results


def main() -> None:
    model = load_model()
    all_results = {
        "knn_agreement": knn_agreement_probe(model),
        "context_size_scaling": context_size_probe(model),
        "contiguous_vs_random_gap": contiguous_vs_random_gap_from_existing_results(),
        "label_permutation": label_permutation_probe(model),
    }
    with open("results/rq3_mechanism_probes.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/rq3_mechanism_probes.json")


if __name__ == "__main__":
    main()
