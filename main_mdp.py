from mdp.mdp import value_iteration, best_policy
from environment.campus_mdp import CampusMDP
from environment.campus_map import build_campus_map
from visualization.plot_mdp import plot_mdp_all_floors

# MDP CONFIGURATION
MDP_CONFIG = {
    "gamma": 0.95,
    "goal_reward": 100.0,

    # Cost scale (tuned)
    "step_cost": 0.2,
    "elevator_cost": 0.5,
    "stair_cost": 0.7,
    "escalator_cost": 0.3,
    "default_state_reward": 0.0,

    # Stochasticity
    "crowd_stay_prob": 0.05,
    "slip_prob": 0.10,
    "elevator_delay_prob": 0.20,
    "escalator_fail_prob": 0.40,
}

def build_campus_mdp(campus_map, goal):
    return CampusMDP(
        campus_map=campus_map,
        terminals=[goal],
        **MDP_CONFIG
    )

def sample_states(start, goal):
    return [
        start,
        (0, 4, 1),   # E
        (0, 11, 1),  # X
        (0, 10, 4),  # A
        (1, 11, 4),  # L
        (4, 3, 7),   # F
        goal,
    ]

def main():
    campus_map = build_campus_map()

    start = (0, 1, 1)
    goal  = (5, 20, 8)

    mdp = build_campus_mdp(campus_map, goal)

    U = value_iteration(mdp, epsilon=0.01)
    pi = best_policy(mdp, U)

    print("\n--- Policy samples (state -> action) ---")
    for s in sample_states(start, goal):
        print(s, "->", pi.get(s, None))

    print("\nUtility(start):", U.get(start, None))
    print("Utility(goal):", U.get(goal, None))

    plot_mdp_all_floors(
        campus_map, U, pi,
        floors=[0, 5],
        title_prefix="MDP Utility + Policy"
    )


if __name__ == "__main__":
    main()
