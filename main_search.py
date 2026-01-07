from search.ucs import uniform_cost_search
from search.bfs import breadth_first_search
from search.astar import astar_search
from heuristics.cost_aware import make_cost_aware_heuristic

from search.common import failure
from environment.campus_search import CampusProblem
from environment.campus_map import build_campus_map
from environment.poi import CATEGORY_TO_CELL, find_cells
from visualization.plot_path import plot_path_all_floors


def reconstruct_path(goal_node):
    """Node -> [state,...]. If failure/None, return None."""
    if goal_node is None or goal_node is failure:
        return None

    path = []
    node = goal_node
    while node is not None:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path


def compute_path_cost(problem, path):
    """Compute path cost using problem.action_cost(s, a, s1)."""
    if not path or len(path) < 2:
        return 0

    cost = 0
    for i in range(len(path) - 1):
        s1 = path[i]
        s2 = path[i + 1]

        found = False
        for action in problem.actions(s1):
            if problem.result(s1, action) == s2:
                cost += problem.action_cost(s1, action, s2)
                found = True
                break

        if not found:
            raise ValueError(f"Invalid transition in path: {s1} -> {s2}")

    return cost


def normalize_category(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def pick_destination_category():
    print("\nWhere do you want to go?")
    print("Options:", ", ".join(sorted(CATEGORY_TO_CELL.keys())))
    user = input("Type one option: ")
    return normalize_category(user)


def main():
    campus_map = build_campus_map()
    initial = (0, 21, 7)

    # 1) Destination category -> goals list
    category = pick_destination_category()
    if category not in CATEGORY_TO_CELL:
        print(f"Unknown category: {category}")
        return

    cell_char = CATEGORY_TO_CELL[category]
    goals = find_cells(campus_map, cell_char)

    if not goals:
        print(f"No cells found for category '{category}' (cell='{cell_char}').")
        return

    # 2) Build multi-goal problem (ANY goal is acceptable)
    problem = CampusProblem(
        initial=initial,
        campus_map=campus_map,
        goals=goals
    )

    # 3) Cost-aware heuristic for A* (single/total cost model)
    # Lower bounds for your fixed costs in CampusProblem:
    # - MOVE cost lower bound = 1
    # - cheapest vertical move lower bound = elevator cost = 2
    h = make_cost_aware_heuristic(
        goals=goals,
        campus_map=campus_map,
        step_cost=1,
        min_vertical_cost=2
    )

    # 4) Run searches
    bfs_goal, bfs_reached = breadth_first_search(problem)
    bfs_path = reconstruct_path(bfs_goal)

    ucs_goal, ucs_reached = uniform_cost_search(problem)
    ucs_path = reconstruct_path(ucs_goal)

    astar_goal, astar_reached = astar_search(problem, h=h)
    astar_path = reconstruct_path(astar_goal)

    # 5) Plots
    if bfs_path:
        plot_path_all_floors(campus_map, bfs_path, title_prefix=f"BFS to nearest {category}")
    if ucs_path:
        plot_path_all_floors(campus_map, ucs_path, title_prefix=f"UCS to nearest {category}")
    if astar_path:
        plot_path_all_floors(campus_map, astar_path, title_prefix=f"A* to nearest {category}")

    # 6) Print results
    print("\n=== Destination ===")
    print("Category:", category, "| Cell:", cell_char)
    print("Start   :", initial)

    print("\n--- BFS (min steps) ---")
    print("Found:", bfs_path is not None)
    print("Reached states:", len(bfs_reached) if bfs_reached is not None else 0)
    if bfs_path:
        print("Steps :", len(bfs_path) - 1)
        print("Cost  :", compute_path_cost(problem, bfs_path))
        print("Goal  :", bfs_path[-1])
        print("Path  :", bfs_path)

    print("\n--- UCS (min cost) ---")
    print("Found:", ucs_path is not None)
    print("Reached states:", len(ucs_reached) if ucs_reached is not None else 0)
    if ucs_path:
        print("Steps :", len(ucs_path) - 1)
        print("Cost  :", ucs_goal.path_cost)
        print("Goal  :", ucs_path[-1])
        print("Path  :", ucs_path)

    print("\n--- A* (min cost + heuristic) ---")
    print("Found:", astar_path is not None)
    print("Reached states:", len(astar_reached) if astar_reached is not None else 0)
    if astar_path:
        print("Steps :", len(astar_path) - 1)
        print("Cost  :", astar_goal.path_cost)
        print("Goal  :", astar_path[-1])
        print("Path  :", astar_path)


if __name__ == "__main__":
    main()
