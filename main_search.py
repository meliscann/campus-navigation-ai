from search.search import breadth_first_search, uniform_cost_search, astar_search
from environment.campus_search import CampusProblem
from environment.campus_map import build_campus_map
from visualization.plot_path import plot_path_all_floors

def build_floor_from_strings(lines):
    return [list(row) for row in lines]

def reconstruct_path(goal_node):
    if goal_node is None:
        return None
    path = []
    node = goal_node
    while node is not None:
        path.append(node.state)
        node = node.parent
    path.reverse()
    return path

def main():
    campus_map = build_campus_map()

    initial = (0, 21, 7)
    goal = (5, 4, 1)

    problem = CampusProblem(initial=initial, goal=goal, campus_map=campus_map)

    bfs_goal = breadth_first_search(problem)
    ucs_goal = uniform_cost_search(problem)

    def h(node):
        f, x, y = node.state
        gf, gx, gy = goal
        return abs(x - gx) + abs(y - gy) + 2 * abs(f - gf)

    astar_goal = astar_search(problem, h=h)

    bfs_path = reconstruct_path(bfs_goal)
    ucs_path = reconstruct_path(ucs_goal)
    astar_path = reconstruct_path(astar_goal)

    plot_path_all_floors(campus_map, bfs_path, title_prefix="BFS Path")
    plot_path_all_floors(campus_map, ucs_path, title_prefix="UCS Path")
    plot_path_all_floors(campus_map, astar_path, title_prefix="A* Path")


    def compute_path_cost(problem, path):
        cost = 0
        for i in range(len(path) - 1):
            s1 = path[i]
            s2 = path[i + 1]

            # action'ı bul (s1 -> s2)
            for action in problem.actions(s1):
                if problem.result(s1, action) == s2:
                    cost = problem.path_cost(cost, s1, action, s2)
                    break
        return cost


    print("\n--- BFS ---")
    print("Found:", bfs_path is not None)
    if bfs_path:
        print("Steps:", len(bfs_path) - 1)
        bfs_cost = compute_path_cost(problem, bfs_path)
        print("Cost :", bfs_cost)
        print("Path :", bfs_path)


    print("\n--- UCS ---")
    print("Found:", ucs_path is not None)
    if ucs_path:
        print("Steps:", len(ucs_path) - 1)
        print("Cost :", ucs_goal.path_cost)
        print("Path :", ucs_path)

    print("\n--- A* ---")
    print("Found:", astar_path is not None)
    if astar_path:
        print("Steps:", len(astar_path) - 1)
        print("Cost :", astar_goal.path_cost)
        print("Path :", astar_path)

if __name__ == "__main__":
    main()



