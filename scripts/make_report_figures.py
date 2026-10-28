"""Generates the figures for the Phase 0-2 report from already-saved
results JSON (checkpoints/history.json, results/*.json) -- no new
evaluation runs, purely a presentation layer over existing data.

Color convention, held fixed across every figure (per the project's
dataviz-skill-validated categorical palette, light-mode hex values, used
in their documented adjacent-safe order):
  - control_prior_model: blue  #2a78d6  (the headline series, always first)
  - knn:                 orange #eb6834 (the central comparison baseline)
  - logistic_regression: aqua   #1baf7a
  - gbt:                 yellow #eda100
  - small_mlp:           magenta #e87ba4
  - majority / "expected if independent" / chance-level references: muted
    gray, dashed -- these are floors/nulls, not comparable series, so they
    deliberately don't consume a categorical slot.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

PREDICTOR_COLORS = {
    "control_prior_model": "#2a78d6",
    "knn": "#eb6834",
    "logistic_regression": "#1baf7a",
    "gbt": "#eda100",
    "small_mlp": "#e87ba4",
}
MAJORITY_COLOR = "#9a9a94"
GENERATOR_COLORS = {"mdp": "#2a78d6", "bandit": "#eb6834", "lds": "#1baf7a"}
PREDICTOR_LABELS = {
    "control_prior_model": "control-prior model",
    "knn": "kNN",
    "logistic_regression": "logistic regression",
    "gbt": "gradient-boosted trees",
    "small_mlp": "small MLP",
    "majority": "majority class",
}
FIGSIZE_SINGLE = (7, 4.5)
DPI = 150


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e0e0dc", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def fig1_training_curve():
    with open("checkpoints/history.json") as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    for generator, color in GENERATOR_COLORS.items():
        steps = [r["step"] for r in history if r["generator"] == generator]
        accs = [r["train_accuracy"] for r in history if r["generator"] == generator]
        ax.scatter(steps, accs, s=10, color=color, alpha=0.55, label=generator, linewidths=0)

    ax.axvline(8800, color="#9a9a94", linestyle="--", linewidth=1.2)
    ax.text(8800, 0.97, "first decision-gate check", fontsize=8, color="#52514e", ha="center", va="top")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("training step")
    ax.set_ylabel("per-batch train accuracy")
    ax.set_title("Training accuracy by generator: LDS/Bandit saturate, MDP plateaus low")
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("results/figures/fig1_training_curve.png", bbox_inches="tight")
    plt.close(fig)


def fig2_rq1_final_comparison():
    with open("results/phase1_rq1_decision_gate.json") as f:
        data = json.load(f)

    systems = list(data.keys())
    predictor_order = ["majority", "knn", "logistic_regression", "gbt", "small_mlp", "control_prior_model"]

    for split in ["contiguous", "random"]:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=DPI)
        n_predictors = len(predictor_order)
        bar_width = 0.8 / n_predictors
        x = np.arange(len(systems))

        for i, predictor in enumerate(predictor_order):
            accs = [data[s][split][predictor]["aggregate"]["accuracy"]["mean"] for s in systems]
            color = MAJORITY_COLOR if predictor == "majority" else PREDICTOR_COLORS[predictor]
            offset = (i - (n_predictors - 1) / 2) * bar_width
            ax.bar(x + offset, accs, width=bar_width, color=color, label=PREDICTOR_LABELS[predictor], zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(systems)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("accuracy")
        split_label = "contiguous (primary protocol)" if split == "contiguous" else "random (leakage control)"
        ax.set_title(f"Final comparison at step 16,800 -- {split_label} split")
        _style_axes(ax)
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(f"results/figures/fig2_rq1_final_comparison_{split}.png", bbox_inches="tight")
        plt.close(fig)


def fig3_rq1_progression():
    # The 800-step and 8800-step intermediate results were printed but only
    # the final (16800-step) run was saved to its own canonical file; the
    # 800/8800 numbers are transcribed here from the commit history
    # (commits a870d28 and dded217) for this progression plot specifically.
    progression = {
        800: {
            "pendulum": {"contiguous": -0.248, "random": -0.228},
            "cartpole": {"contiguous": -0.212, "random": -0.282},
            "dubins": {"contiguous": -0.129, "random": -0.263},
        },
        8800: {
            "pendulum": {"contiguous": -0.016, "random": -0.013},
            "cartpole": {"contiguous": -0.007, "random": -0.050},
            "dubins": {"contiguous": -0.034, "random": -0.069},
        },
        16800: {
            "pendulum": {"contiguous": -0.011, "random": -0.016},
            "cartpole": {"contiguous": -0.007, "random": -0.049},
            "dubins": {"contiguous": -0.041, "random": -0.060},
        },
    }
    steps = sorted(progression.keys())
    system_colors = {"pendulum": "#2a78d6", "cartpole": "#eb6834", "dubins": "#1baf7a"}

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    for system, color in system_colors.items():
        for split, linestyle, marker in [("contiguous", "-", "o"), ("random", "--", "s")]:
            values = [progression[s][system][split] for s in steps]
            label = f"{system} ({split})"
            ax.plot(steps, values, color=color, linestyle=linestyle, marker=marker, markersize=5, linewidth=1.8, label=label)

    ax.axhline(0, color="#9a9a94", linewidth=1, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(steps)
    ax.set_xticklabels([str(s) for s in steps])
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("model accuracy - best baseline accuracy")
    ax.set_title("The gap closes fast, then plateaus (model never exceeds 0)")
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig("results/figures/fig3_rq1_progression.png", bbox_inches="tight")
    plt.close(fig)


def fig4_rq2_dose_response():
    with open("results/rq2_autocorrelation_dose_response.json") as f:
        data = json.load(f)["rq2_dose_response"]

    decay_rates = sorted(float(k) for k in data.keys())
    predictor_order = ["majority", "knn", "logistic_regression", "gbt", "small_mlp", "control_prior_model"]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE, dpi=DPI)
    for predictor in predictor_order:
        accs = [data[str(dr)][predictor]["accuracy"] for dr in decay_rates]
        color = MAJORITY_COLOR if predictor == "majority" else PREDICTOR_COLORS[predictor]
        linestyle = "--" if predictor == "majority" else "-"
        linewidth = 2.4 if predictor == "control_prior_model" else 1.6
        ax.plot(decay_rates, accs, color=color, linestyle=linestyle, linewidth=linewidth, marker="o", markersize=4, label=PREDICTOR_LABELS[predictor])

    ax.set_xlabel("LDS decay rate (higher = more autocorrelated)")
    ax.set_ylabel("accuracy")
    ax.set_title("Model's dose-response to autocorrelation tracks kNN's (r=0.991)")
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/fig4_rq2_dose_response.png", bbox_inches="tight")
    plt.close(fig)


def fig5_rq3_context_scaling():
    with open("results/rq3_mechanism_probes.json") as f:
        data = json.load(f)["context_size_scaling"]

    systems = list(data.keys())
    fig, axes = plt.subplots(1, len(systems), figsize=(13, 4), dpi=DPI, sharey=True)
    for ax, system in zip(axes, systems):
        context_sizes = sorted(int(c) for c in data[system].keys())
        model_accs = [data[system][str(c)]["model_accuracy"] for c in context_sizes]
        knn_accs = [data[system][str(c)]["knn_accuracy"] for c in context_sizes]

        ax.plot(context_sizes, model_accs, color=PREDICTOR_COLORS["control_prior_model"], linewidth=2.2, marker="o", markersize=4, label="model")
        ax.plot(context_sizes, knn_accs, color=PREDICTOR_COLORS["knn"], linewidth=1.8, marker="s", markersize=4, label="kNN")
        ax.set_title(system)
        ax.set_xlabel("context size")
        _style_axes(ax)

    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0.4, 0.9)
    axes[-1].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Context-size scaling: model vs. kNN accuracy (see report text for per-system reading)", y=1.02)
    fig.tight_layout()
    fig.savefig("results/figures/fig5_rq3_context_scaling.png", bbox_inches="tight")
    plt.close(fig)


def fig6_rq3_knn_agreement():
    with open("results/rq3_mechanism_probes.json") as f:
        data = json.load(f)["knn_agreement"]

    systems = list(data.keys())
    observed = [data[s]["observed_agreement"] for s in systems]
    expected = [data[s]["expected_if_independent"] for s in systems]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=DPI)
    x = np.arange(len(systems))
    width = 0.32
    ax.bar(x - width / 2, observed, width=width, color=PREDICTOR_COLORS["control_prior_model"], label="observed agreement", zorder=2)
    ax.bar(x + width / 2, expected, width=width, color=MAJORITY_COLOR, label="expected if independent", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Model and kNN agree far more than chance given their accuracies")
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig("results/figures/fig6_rq3_knn_agreement.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_training_curve()
    fig2_rq1_final_comparison()
    fig3_rq1_progression()
    fig4_rq2_dose_response()
    fig5_rq3_context_scaling()
    fig6_rq3_knn_agreement()
    print("Saved 7 figures to results/figures/")
