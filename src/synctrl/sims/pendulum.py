"""Inverted pendulum swing-up with a discrete-torque energy-pumping expert.

State: [theta, theta_dot], theta = 0 is the pendulum hanging straight down,
theta = pi is upright (the swing-up target). Dynamics are the standard damped
pendulum under an applied torque, integrated with semi-implicit Euler (stable
for oscillatory systems at the timesteps used here, unlike explicit Euler).

Expert policy: energy-shaping swing-up, a well-known closed-form controller
for this system. Applying torque with the same sign as theta_dot * cos(theta)
monotonically pumps mechanical energy toward the upright equilibrium's energy
level; near the top we switch to a linear balancing rule. Both branches are
bang-bang over the discrete action set {-1, 0, +1} (scaled by max_torque).
"""

from __future__ import annotations

import numpy as np

from .base import KnownExpertSystem

CONFIG = {
    # Physical parameter ranges, randomized per-rollout so the generator sees
    # a family of systems rather than one fixed pendulum.
    "mass_range": (0.5, 2.0),  # kg; broad enough to vary swing-up torque needs
    "length_range": (0.3, 1.0),  # m
    "damping_range": (0.0, 0.05),  # N*m*s/rad; light damping, physically plausible
    "gravity": 9.81,  # m/s^2; fixed, not a parameter we want the model to have to infer
    # Actuator limit is a multiple of m*g*l (the gravity torque at horizontal)
    # rather than a fixed N*m value, so control authority scales with the
    # randomized mass/length instead of becoming infeasible for heavy/long
    # draws. Linearizing near upright, the destabilizing gravity term is
    # m*g*l*angle_error within the balance band (angle_error < 0.5 rad), so a
    # bang-bang torque of >0.5x m*g*l is needed just to dominate it; 1.5x
    # leaves comfortable margin for both balance and a fast swing-up.
    "max_torque_factor": 1.5,
    "dt": 0.02,  # s; 50 Hz, typical for this class of control problem
    # Angle (rad) within which we switch from energy-pumping to the balancing rule.
    "balance_angle_band": 0.5,
    # Balancing rule gains (proportional-derivative on angle error from upright).
    "balance_kp": 3.0,
    "balance_kd": 0.5,
}


def _wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


class Pendulum(KnownExpertSystem):
    action_space_size = 3  # {-1, 0, +1} x max_torque
    state_dim = 2

    def __init__(self, config: dict = CONFIG):
        self.config = config

    def sample_params(self, rng: np.random.Generator) -> dict:
        c = self.config
        return {
            "mass": rng.uniform(*c["mass_range"]),
            "length": rng.uniform(*c["length_range"]),
            "damping": rng.uniform(*c["damping_range"]),
        }

    def sample_initial_state(self, rng: np.random.Generator, params: dict) -> np.ndarray:
        # Start hanging near the bottom with small random perturbation: the
        # swing-up controller then has real work to do.
        theta = np.pi + rng.uniform(-0.2, 0.2)
        theta_dot = rng.uniform(-0.5, 0.5)
        return np.array([_wrap_angle(theta), theta_dot])

    def step(self, state: np.ndarray, action: int, params: dict) -> np.ndarray:
        c = self.config
        theta, theta_dot = state
        m, l, b, g = params["mass"], params["length"], params["damping"], c["gravity"]
        max_torque = c["max_torque_factor"] * m * g * l
        torque = (action - 1) * max_torque  # action index 0,1,2 -> -1,0,+1
        # Point-mass pendulum: I * theta_ddot = -b*theta_dot - m*g*l*sin(theta) + torque,
        # with theta=0 defined as hanging down (so gravity restores toward theta=0).
        inertia = m * l ** 2
        theta_ddot = (torque - b * theta_dot - m * g * l * np.sin(theta)) / inertia

        # Semi-implicit (symplectic) Euler: update velocity first, then use the
        # new velocity for the position update. Far more stable than explicit
        # Euler for oscillatory dynamics at this dt.
        new_theta_dot = theta_dot + c["dt"] * theta_ddot
        new_theta = _wrap_angle(theta + c["dt"] * new_theta_dot)
        return np.array([new_theta, new_theta_dot])

    def expert_action(self, state: np.ndarray, params: dict) -> int:
        c = self.config
        theta, theta_dot = state
        angle_from_upright = _wrap_angle(theta - np.pi)

        if abs(angle_from_upright) < c["balance_angle_band"]:
            # Near upright: PD control on the (small) angle error, bang-bang
            # over the discrete action set.
            u = -(c["balance_kp"] * angle_from_upright + c["balance_kd"] * theta_dot)
        else:
            # Energy-pumping swing-up: push in the direction that increases
            # mechanical energy, i.e. same sign as theta_dot * cos(theta).
            u = np.sign(theta_dot * np.cos(theta))
            if u == 0:
                u = 1.0  # break the tie at the unstable equilibrium theta_dot=0

        if u > 1e-6:
            return 2  # +1
        elif u < -1e-6:
            return 0  # -1
        return 1  # 0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sys = Pendulum()

    rollout = sys.rollout(n_steps=500, rng=rng, noise_eps=0.0)
    final_angle_from_upright = _wrap_angle(rollout.states[-1, 0] - np.pi)
    print(f"params: {rollout.params}")
    print(f"final theta (rad from upright): {final_angle_from_upright:.4f}")
    print(f"final theta_dot: {rollout.states[-1, 1]:.4f}")
    assert abs(final_angle_from_upright) < 0.3, "swing-up controller failed to reach upright"
    print("sanity check passed: deterministic swing-up reaches upright.")

    # Noisy rollout: empirical action-match rate should track the analytic ceiling.
    noise_eps = 0.2
    n_trials = 20000
    matches = 0
    for _ in range(n_trials):
        s = sys.sample_initial_state(rng, sys.sample_params(rng))
        p = sys.sample_params(rng)
        expert_a = sys.expert_action(s, p)
        noisy_a = expert_a if rng.random() >= noise_eps else int(rng.integers(0, 3))
        matches += int(noisy_a == expert_a)
    empirical = matches / n_trials
    analytic = sys.oracle_accuracy_ceiling(noise_eps)
    print(f"empirical match rate: {empirical:.4f}, analytic ceiling: {analytic:.4f}")
    assert abs(empirical - analytic) < 0.02, "empirical noise model doesn't match analytic ceiling"
    print("sanity check passed: noise model matches analytic oracle ceiling.")
