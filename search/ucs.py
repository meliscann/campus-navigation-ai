from .common import best_first_search, g

def uniform_cost_search(problem):
    """
    Uniform-cost search (UCS).

    Returns:
      (solution_node, reached_dict)
    """
    return best_first_search(problem, f=g)
