import heapq
import math
from collections import defaultdict, deque

# 1) PROBLEM and NODE CLASSES
class Problem(object):
    """
    Abstract class for a formal problem.
    A new domain subclasses this, overriding `actions` and `result`.
    Default heuristic is 0 and default action cost is 1 for all transitions.
    """
    def __init__(self, initial=None, goal=None, **kwds):
        self.initial = initial
        self.goal = goal
        self.__dict__.update(**kwds)

    def actions(self, state):        raise NotImplementedError
    def result(self, state, action): raise NotImplementedError
    def is_goal(self, state):        return state == self.goal
    def action_cost(self, s, a, s1): return 1
    def h(self, node):               return 0

    def __str__(self):
        return '{}({!r}, {!r})'.format(type(self).__name__, self.initial, self.goal)


class Node:
    """A Node in a search tree."""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self):
        return '<{}>'.format(self.state)

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


failure = Node('failure', path_cost=math.inf)
cutoff  = Node('cutoff',  path_cost=math.inf)


# 2) CORE HELPERS
def expand(problem, node):
    """Expand a node, generating the child nodes."""
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node):
    """The sequence of actions to get to this node."""
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    """The sequence of states to get to this node."""
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


def is_cycle(node):
    """Checks whether a node repeats a previously encountered state on its path."""
    state = node.state
    while node.parent:
        if node.parent.state == state:
            return True
        node = node.parent
    return False


# 3) QUEUES
FIFOQueue = deque
LIFOQueue = list

class PriorityQueue:
    """A queue in which the item with minimum key(item) is popped first."""
    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []
        for item in items:
            self.add(item)

    def add(self, item):
        heapq.heappush(self.items, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self.items)[1]

    def top(self):
        return self.items[0][1]

    def __len__(self):
        return len(self.items)


# 4) BEST-FIRST SEARCH (UCS & A*)
def best_first_search(problem, f):
    """
    Search nodes with minimum f(node) value first.

    Returns:
      (solution_node, reached_dict)
        - reached_dict: state -> best Node found for that state
    """
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}

    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node, reached

        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)

    return failure, reached

def g(n):
    return n.path_cost

# 5) OPTIONAL GENERIC MAP / GRID CLASSES
def straight_line_distance(A, B):
    """Straight-line distance between two points."""
    return sum(abs(a - b) ** 2 for (a, b) in zip(A, B)) ** 0.5


def multimap(pairs) -> dict:
    """Given (key, val) pairs, make a dict of {key: [val,...]}."""
    result = defaultdict(list)
    for key, val in pairs:
        result[key].append(val)
    return result


class Map:
    """A map of places in a 2D world: a graph with vertices and weighted links."""
    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'):
            links = {link: 1 for link in links}
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links.items())
        self.locations = locations or defaultdict(lambda: (0, 0))


class RouteProblem(Problem):
    """A problem to find a route between locations on a Map."""
    def actions(self, state):
        return self.map.neighbors[state]

    def result(self, state, action):
        return action if action in self.map.neighbors[state] else state

    def action_cost(self, s, action, s1):
        return self.map.distances[s, s1]

    def h(self, node):
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])


class GridProblem(Problem):
    """Finding a path on a 2D grid with obstacles."""
    directions = [(-1, -1), (0, -1), (1, -1),
                  (-1, 0),           (1,  0),
                  (-1, +1), (0, +1), (1, +1)]

    def __init__(self, initial=(15, 30), goal=(130, 30), obstacles=(), **kwds):
        kwds['initial'] = initial
        kwds['goal'] = goal
        kwds['obstacles'] = set(obstacles) - {initial, goal}
        super().__init__(**kwds)

    def action_cost(self, s, action, s1):
        return straight_line_distance(s, s1)

    def h(self, node):
        return straight_line_distance(node.state, self.goal)

    def result(self, state, action):
        return action if action not in self.obstacles else state

    def actions(self, state):
        x, y = state
        possible_moves = set((x + dx, y + dy) for dx, dy in self.directions)
        return possible_moves - self.obstacles
