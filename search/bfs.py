from .common import Node, FIFOQueue, expand, failure

def breadth_first_search(problem):
    """
    Search shallowest nodes first (BFS).

    Returns:
      (solution_node, reached_set)
    """
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node, {problem.initial}

    frontier = FIFOQueue([node])
    reached = {problem.initial}

    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                reached.add(s)
                return child, reached
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)

    return failure, reached
