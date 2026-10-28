"""RQ2 (docs/00_critique_and_plan.md): is the effect explained by temporal
autocorrelation/local smoothness specifically?

Directly manipulates the one parameter the LDS generator's decay rate
controls -- how autocorrelated the state trajectory is -- holding
everything else about the generator fixed, and measures how the trained
control-prior model's accuracy tracks it, compared against kNN (the most
literal "local smoothing" baseline) and the rest of the baseline suite.

This is a *falsifiable* dose-response test: if the model's zero-shot
accuracy rises and falls in lockstep with kNN's as autocorrelation varies,
that's direct evidence its behavior is well explained by local smoothing,
not learned long-horizon control structure.

Important limitation, found while looking at the actual numbers rather than
assumed up front: decay_rate=0 is NOT a clean "no exploitable structure"
control. At decay_rate=0, x_{t+1} = b_action + noise (the state has zero
memory of its own past), but this still leaves a strong *static*
state-action association -- each action's fixed displacement b_a gives it
a characteristic state "signature" that kNN, logistic regression, or the
model can all exploit without needing any temporal memory at all (this is
structurally closer to the Bandit generator than to "no structure"). So
this sweep varies autocorrelation *given that a static association is
always present*, not autocorrelation against a true null -- see the
analysis after the sweep for what that does and doesn't let us conclude.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from synctrl.baselines import ALL_BASELINES
from synctrl.eval.harness import aggregate, as_predict_fn, evaluate_on_tasks
from synctrl.generators import lds
from synctrl.model import ControlPriorModel, build_model
from synctrl.tasks import task_from_arrays

DECAY_RATE_SWEEP = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]
CONFIG = {
    "n_tasks_per_point": 80,
    "seq_len": 200,
    "num_features": 20,  # matches the model's training-time num_features
    "max_actions": 10,  # matches the model's training-time max_actions
    "seed": 0,
}


def load_model(checkpoint_path: str = "checkpoints/step_16800.pt") -> ControlPriorModel:
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
        rng=np.random.default_rng(0),
    )


def tasks_at_fixed_decay_rate(decay_rate: float, rng: np.random.Generator) -> list:
    config = {**lds.CONFIG, "decay_rate_range": (decay_rate, decay_rate)}
    x, y, _ = lds.get_batch(
        batch_size=CONFIG["n_tasks_per_point"], seq_len=CONFIG["seq_len"],
        num_features=CONFIG["num_features"], rng=rng, max_actions=CONFIG["max_actions"], config=config,
    )
    # task_from_arrays casts y to int64 internally -- get_batch's TabPFN
    # convention returns float32 labels throughout.
    x_np, y_np = x.numpy(), y.numpy()  # (seq_len, batch, num_features) / (seq_len, batch)

    tasks = []
    for b in range(CONFIG["n_tasks_per_point"]):
        tasks.append(
            task_from_arrays(
                x_np[:, b, :], y_np[:, b], action_space_size=CONFIG["max_actions"],
                split="contiguous", rng=rng, context_frac=0.5,
                system_params={"decay_rate": decay_rate},
            )
        )
    return tasks


def main() -> None:
    model_predictor = load_model()
    predictors = ALL_BASELINES + [model_predictor]
    rng = np.random.default_rng(CONFIG["seed"])

    results = {}
    print(f"{'decay_rate':<12}" + "".join(f"{p.name:<22}" for p in predictors))
    for decay_rate in DECAY_RATE_SWEEP:
        tasks = tasks_at_fixed_decay_rate(decay_rate, rng)
        row = {}
        for predictor in predictors:
            agg_rng = np.random.default_rng(CONFIG["seed"])
            records = evaluate_on_tasks(as_predict_fn(predictor), tasks)
            row[predictor.name] = aggregate(records, agg_rng)
        results[decay_rate] = row
        print(f"{decay_rate:<12.3f}" + "".join(f"{row[p.name].accuracy.mean:<22.3f}" for p in predictors))

    jsonable = {
        str(decay_rate): {name: {"accuracy": agg.accuracy.mean, "nll": agg.nll.mean} for name, agg in row.items()}
        for decay_rate, row in results.items()
    }
    with open("results/rq2_autocorrelation_dose_response.json", "w") as f:
        json.dump({"rq2_dose_response": jsonable}, f, indent=2)

    print("\n--- Does the model's dose-response curve track kNN's? ---")
    model_accs = np.array([results[d]["control_prior_model"].accuracy.mean for d in DECAY_RATE_SWEEP])
    knn_accs = np.array([results[d]["knn"].accuracy.mean for d in DECAY_RATE_SWEEP])
    correlation = np.corrcoef(model_accs, knn_accs)[0, 1]
    print(f"model accuracy across decay rates: {np.round(model_accs, 3)}")
    print(f"knn   accuracy across decay rates: {np.round(knn_accs, 3)}")
    print(f"Pearson correlation between model's and kNN's dose-response curves: {correlation:.3f}")
    low_autocorr_gap = model_accs[0] - knn_accs[0]
    print(
        f"model - kNN at decay_rate=0.0 (zero temporal memory, but still a static state-action "
        f"association from each action's fixed displacement -- see module docstring): {low_autocorr_gap:+.3f}"
    )


if __name__ == "__main__":
    main()
