"""Dubins car navigating to a fixed goal with a proportional heading expert.

State: [x, y, heading]. The car moves at constant speed and can only steer
(bang-bang left/straight/right), the classic Dubins-car model. The goal
position is a randomized parameter, not part of the evolving state, since
it is fixed for the duration of one rollout.

Expert policy: proportional heading control toward the goal (a restricted
form of pure pursuit) -- turn toward the sign of the bearing error, go
straight once aligned within a deadband. This greedily reduces heading
error at every step; it does not claim to be the time-optimal Dubins path,
only a fixed, known reference policy.

Because the car moves at constant speed and cannot stop, this controller
converges to *orbiting* the goal at roughly radius speed/turn_rate rather
than coming to rest on it -- "success" is therefore checked as the minimum
distance to the goal *over the whole trajectory*, not the final-step
distance, which would spuriously fail if the car flies past the goal and
is mid-orbit at the moment the rollout happens to end.
"""

from __future__ import annotations

import numpy as np

from .base import KnownExpertSystem

CONFIG = {
    "speed_range": (0.5, 1.5),  # m/s
    # A Dubins car can't stop, so a heading-pursuit controller converges to
    # *orbiting* the goal at roughly speed/turn_rate, not resting on it. Lower
    # bound kept well above speed's upper bound so the worst-case orbit
    # radius (1.5/1.0 = 1.5 m) stays comfortably inside the "reached the
    # goal's vicinity" check used in tests and the sanity script below.
    "turn_rate_range": (1.0, 2.5),  # rad/s, applied at full +-1 steering
    "goal_radius_range": (3.0, 8.0),  # m, goal sampled on a circle of this radius
    "dt": 0.05,  # s; 20 Hz, coarser than the pendulum/cartpole since the car is slower
    # Heading-error deadband (rad) within which the expert goes straight
    # instead of turning; avoids chattering once aligned with the goal.
    "heading_deadband": 0.05,
}


def _wrap_angle(theta: float) -> float:
    return (theta + np.pi) % (2 * np.pi) - np.pi


class DubinsCar(KnownExpertSystem):
    action_space_size = 3  # steer {-1, 0, +1} x turn_rate
    state_dim = 3

    def __init__(self, config: dict = CONFIG):
        self.config = config

    def sample_params(self, rng: np.random.Generator) -> dict:
        c = self.config
        goal_angle = rng.uniform(-np.pi, np.pi)
        goal_radius = rng.uniform(*c["goal_radius_range"])
        return {
            "speed": rng.uniform(*c["speed_range"]),
            "turn_rate": rng.uniform(*c["turn_rate_range"]),
            "goal_x": goal_radius * np.cos(goal_angle),
            "goal_y": goal_radius * np.sin(goal_angle),
        }

    def sample_initial_state(self, rng: np.random.Generator, params: dict) -> np.ndarray:
        heading = rng.uniform(-np.pi, np.pi)
        return np.array([0.0, 0.0, heading])

    def step(self, state: np.ndarray, action: int, params: dict) -> np.ndarray:
        c = self.config
        x, y, heading = state
        steer = action - 1  # action index 0,1,2 -> -1,0,+1

        new_heading = _wrap_angle(heading + c["dt"] * steer * params["turn_rate"])
        new_x = x + c["dt"] * params["speed"] * np.cos(heading)
        new_y = y + c["dt"] * params["speed"] * np.sin(heading)
        return np.array([new_x, new_y, new_heading])

    def expert_action(self, state: np.ndarray, params: dict) -> int:
        c = self.config
        x, y, heading = state
        bearing_to_goal = np.arctan2(params["goal_y"] - y, params["goal_x"] - x)
        heading_error = _wrap_angle(bearing_to_goal - heading)

        if heading_error > c["heading_deadband"]:
            return 2  # turn left (+1)
        elif heading_error < -c["heading_deadband"]:
            return 0  # turn right (-1)
        return 1  # straight


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sys = DubinsCar()

    rollout = sys.rollout(n_steps=500, rng=rng, noise_eps=0.0)
    params = rollout.params
    goal_xy = np.array([params["goal_x"], params["goal_y"]])
    dists_to_goal = np.linalg.norm(rollout.states[:, :2] - goal_xy, axis=1)
    min_dist_to_goal = dists_to_goal.min()
    print(f"params: {params}")
    print(f"goal: {goal_xy}")
    print(f"min distance to goal over trajectory: {min_dist_to_goal:.4f} m")
    assert min_dist_to_goal < 2.0, "proportional heading controller failed to reach the goal's vicinity"
    print("sanity check passed: heading controller converges toward the goal.")

    noise_eps = 0.25
    n_trials = 20000
    matches = 0
    for _ in range(n_trials):
        p = sys.sample_params(rng)
        s = sys.sample_initial_state(rng, p)
        expert_a = sys.expert_action(s, p)
        noisy_a = expert_a if rng.random() >= noise_eps else int(rng.integers(0, 3))
        matches += int(noisy_a == expert_a)
    empirical = matches / n_trials
    analytic = sys.oracle_accuracy_ceiling(noise_eps)
    print(f"empirical match rate: {empirical:.4f}, analytic ceiling: {analytic:.4f}")
    assert abs(empirical - analytic) < 0.02, "empirical noise model doesn't match analytic ceiling"
    print("sanity check passed: noise model matches analytic oracle ceiling.")
