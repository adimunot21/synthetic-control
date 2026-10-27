"""Cart-pole balancing with a discrete-force PD bang-bang expert.

State: [x, x_dot, theta, theta_dot], theta = 0 is the pole upright (the
balance target). Dynamics follow the standard Barto-Sutton-Anderson
cart-pole equations, integrated with semi-implicit Euler. Two discrete
actions: push the cart left or right with a fixed-magnitude force (no
"do nothing" action, matching the classic formulation of this task).

Expert policy: push in the direction that counteracts the pole's current
lean and angular velocity (a = +1 if theta + kd * theta_dot > 0 else -1).
This is the standard simple heuristic controller for this task and balances
reliably from small initial deviations; it is a fixed, known reference
policy, not a claim of optimality.
"""

from __future__ import annotations

import numpy as np

from .base import KnownExpertSystem

CONFIG = {
    "cart_mass_range": (0.5, 2.0),  # kg
    "pole_mass_range": (0.05, 0.5),  # kg; much lighter than the cart, as is physical
    "pole_half_length_range": (0.25, 0.75),  # m, distance from pivot to center of mass
    "gravity": 9.81,  # m/s^2; fixed
    "force_magnitude": 10.0,  # N; fixed actuator strength
    "dt": 0.02,  # s; 50 Hz
    # Heuristic controller gain on angular velocity (standard choice for this
    # heuristic: a small positive weight on the rate term stabilizes the bang-bang
    # switching surface near theta=0 instead of chattering only on angle sign).
    "balance_kd": 0.3,
}


class CartPole(KnownExpertSystem):
    action_space_size = 2  # push left (-F) or right (+F), no neutral action
    state_dim = 4

    def __init__(self, config: dict = CONFIG):
        self.config = config

    def sample_params(self, rng: np.random.Generator) -> dict:
        c = self.config
        return {
            "cart_mass": rng.uniform(*c["cart_mass_range"]),
            "pole_mass": rng.uniform(*c["pole_mass_range"]),
            "pole_half_length": rng.uniform(*c["pole_half_length_range"]),
        }

    def sample_initial_state(self, rng: np.random.Generator, params: dict) -> np.ndarray:
        # Small random deviation from upright-at-origin, the standard cart-pole
        # balance start state.
        x = rng.uniform(-0.1, 0.1)
        x_dot = rng.uniform(-0.1, 0.1)
        theta = rng.uniform(-0.1, 0.1)
        theta_dot = rng.uniform(-0.1, 0.1)
        return np.array([x, x_dot, theta, theta_dot])

    def step(self, state: np.ndarray, action: int, params: dict) -> np.ndarray:
        c = self.config
        x, x_dot, theta, theta_dot = state
        force = c["force_magnitude"] * (1.0 if action == 1 else -1.0)

        M, m, l, g = (
            params["cart_mass"],
            params["pole_mass"],
            params["pole_half_length"],
            c["gravity"],
        )
        sin_t, cos_t = np.sin(theta), np.cos(theta)

        theta_ddot = (
            g * sin_t + cos_t * (-force - m * l * theta_dot ** 2 * sin_t) / (M + m)
        ) / (l * (4.0 / 3.0 - m * cos_t ** 2 / (M + m)))
        x_ddot = (force + m * l * (theta_dot ** 2 * sin_t - theta_ddot * cos_t)) / (M + m)

        new_x_dot = x_dot + c["dt"] * x_ddot
        new_theta_dot = theta_dot + c["dt"] * theta_ddot
        new_x = x + c["dt"] * new_x_dot
        new_theta = theta + c["dt"] * new_theta_dot
        return np.array([new_x, new_x_dot, new_theta, new_theta_dot])

    def expert_action(self, state: np.ndarray, params: dict) -> int:
        _, _, theta, theta_dot = state
        c = self.config
        switching_value = theta + c["balance_kd"] * theta_dot
        return 1 if switching_value > 0 else 0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sys = CartPole()

    rollout = sys.rollout(n_steps=500, rng=rng, noise_eps=0.0)
    max_abs_theta = np.max(np.abs(rollout.states[:, 2]))
    print(f"params: {rollout.params}")
    print(f"max |theta| over rollout: {max_abs_theta:.4f} rad")
    assert max_abs_theta < 0.5, "balance controller let the pole fall over"
    print("sanity check passed: PD bang-bang keeps the pole balanced.")

    noise_eps = 0.15
    n_trials = 20000
    matches = 0
    for _ in range(n_trials):
        p = sys.sample_params(rng)
        s = sys.sample_initial_state(rng, p)
        expert_a = sys.expert_action(s, p)
        noisy_a = expert_a if rng.random() >= noise_eps else int(rng.integers(0, 2))
        matches += int(noisy_a == expert_a)
    empirical = matches / n_trials
    analytic = sys.oracle_accuracy_ceiling(noise_eps)
    print(f"empirical match rate: {empirical:.4f}, analytic ceiling: {analytic:.4f}")
    assert abs(empirical - analytic) < 0.02, "empirical noise model doesn't match analytic ceiling"
    print("sanity check passed: noise model matches analytic oracle ceiling.")
