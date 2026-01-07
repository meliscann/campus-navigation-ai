from typing import List, Tuple

State = Tuple[int, int, int]  # (floor, x, y)


def build_vertical_cells_index(campus_map):
    """
    For each floor, collect positions of vertical transfer cells (E, S, X).
    """
    vertical_cells = []
    for f, floor in enumerate(campus_map):
        s = set()
        for y in range(len(floor)):
            for x in range(len(floor[0])):
                if floor[y][x] in ("E", "S", "X"):
                    s.add((x, y))
        vertical_cells.append(s)
    return vertical_cells


def make_cost_aware_heuristic(
    goals: List[State],
    campus_map,
    step_cost: int = 1,
    min_vertical_cost: int = 2,
):
    """
    Admissible, cost-aware heuristic for multi-floor campus navigation.
    """

    vertical_cells = build_vertical_cells_index(campus_map)

    def manhattan_xy(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def dist_to_nearest_vertical(f, x, y):
        cells = vertical_cells[f]
        if not cells:
            return 0
        return min(manhattan_xy(x, y, cx, cy) for (cx, cy) in cells)

    def h(node):
        f, x, y = node.state
        to_vertical = step_cost * dist_to_nearest_vertical(f, x, y)

        best = float("inf")
        for (gf, gx, gy) in goals:
            df = abs(f - gf)
            base_xy = step_cost * manhattan_xy(x, y, gx, gy)

            if df == 0:
                cost_lb = base_xy
            else:
                cost_lb = base_xy + to_vertical + (min_vertical_cost * df)

            best = min(best, cost_lb)

        return best

    return h
