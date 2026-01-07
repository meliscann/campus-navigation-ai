from .common import best_first_search, g

def astar_search(problem, h=None):
    """
    A* search.

    Returns:
      (solution_node, reached_dict)
    """
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))
