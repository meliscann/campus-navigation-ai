from mdp.mdp import value_iteration, best_policy
from environment.campus_mdp import CampusMDP
from environment.campus_map import build_campus_map
from visualization.plot_mdp import plot_mdp_all_floors

def main():
    campus_map = build_campus_map()

    start = (0, 1, 1)
    goal  = (5, 20, 8)

    mdp = CampusMDP(
        campus_map=campus_map,
        terminals=[goal],
        gamma=0.95,
        goal_reward=100,
        crowd_stay_prob=0.05,
        slip_prob=0.10,
        elevator_delay_prob=0.20,
        escalator_fail_prob=0.40
    )

    U = value_iteration(mdp, epsilon=0.01)
    pi = best_policy(mdp, U)

    samples = [
        start,          # başlangıç
        (0, 4, 1),      # E (0. kat asansör)
        (0, 11, 1),     # X (0. kat yürüyen merdiven)
        (0, 10, 4),     # O (0. kat öğrenci işleri)
        (1, 11, 4),     # L (1. kat kütüphane)
        (4, 3, 7),      # F (4. kat fakülte)
        (5, 20, 8),     # T (5. kat teras) -> goal
    ]

    print("\n--- Policy samples (state -> action) ---")
    for s in samples:
        print(s, "->", pi.get(s, None))

    print("\nUtility(start):", U.get(start, None))
    print("Utility(goal):", U.get(goal, None))

    plot_mdp_all_floors(campus_map, U, pi, floors=[0, 5], title_prefix="MDP Utility+Policy")

if __name__ == "__main__":
    main()
