import matplotlib.pyplot as plt
import random
import heapq
import math
import sys
import functools
from collections import defaultdict, deque, Counter
from itertools import combinations

# =============================================================================
# 1. PROBLEM and NODE Classes
# =============================================================================

class Problem(object):
    """
    The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
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
        return '{}({!r}, {!r})'.format(
            type(self).__name__, self.initial, self.goal)

class Node:
    """A Node in a search tree."""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.__dict__.update(state=state, parent=parent, action=action, path_cost=path_cost)

    def __repr__(self): return '<{}>'.format(self.state)
    def __len__(self): return 0 if self.parent is None else (1 + len(self.parent))
    def __lt__(self, other): return self.path_cost < other.path_cost
    
    
failure = Node('failure', path_cost=math.inf) # Indicates an algorithm couldn't find a solution.
cutoff  = Node('cutoff',  path_cost=math.inf) # Indicates iterative deepening search was cut off.
    
    
def expand(problem, node):
    """Expand a node, generating the children nodes."""
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        # action_cost should be called problem.action_cost(s, action, s1).
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
    """Checks whether a node repeats a previously encountered state."""
    state = node.state
    while node.parent:
        if node.parent.state == state:
            return True
        node = node.parent
    return False

# =============================================================================
# 2. DATA STRUCTURES (QUEUES)
# =============================================================================

FIFOQueue = deque

LIFOQueue = list

class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x): 
        self.key = key
        self.items = [] # a heap of (score, item) pairs
        for item in items:
            self.add(item)
         
    def add(self, item):
        """Add item to the queue."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]
    
    def top(self): return self.items[0][1]

    def __len__(self): return len(self.items)

# =============================================================================
# 3. BASIC SEARCH ALGORITHMS
# =============================================================================

def breadth_first_search(problem):
    """Search shallowest nodes in the search tree first."""
    node = Node(problem.initial)
    if problem.is_goal(problem.initial):
        return node
    frontier = FIFOQueue([node])
    reached = {problem.initial}
    while frontier:
        node = frontier.pop()
        for child in expand(problem, node):
            s = child.state
            if problem.is_goal(s):
                return child
            if s not in reached:
                reached.add(s)
                frontier.appendleft(child)
    return failure


def iterative_deepening_search(problem):
    """Do depth-limited search with increasing depth limits."""
    for limit in range(1, sys.maxsize):
        result = depth_limited_search(problem, limit)
        if result != cutoff:
            return result

def depth_limited_search(problem, limit=10):
    """Search deepest nodes in the search tree first."""
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        # is_cycle is not defined here.
        # This function is assumed to be defined in the original notebook.
        elif len(node) >= limit: # or is_cycle(node):
            result = cutoff
        elif not is_cycle(node):
            for child in expand(problem, node):
                frontier.append(child)
    return result


def depth_first_recursive_search(problem, node=None):
    if node is None: 
        node = Node(problem.initial)
    if problem.is_goal(node.state):
        return node
    elif is_cycle(node):
        return failure
    else:
        for child in expand(problem, node):
            result = depth_first_recursive_search(problem, child)
            if result:
                return result
        return failure

# =============================================================================
# 4. MAP, ROUTE and GRID Classes
# =============================================================================

class RouteProblem(Problem):
    """A problem to find a route between locations on a `Map`."""
    
    def actions(self, state): 
        """The places neighboring `state`."""
        return self.map.neighbors[state]
    
    def result(self, state, action):
        """Go to the `action` place, if the map says that is possible."""
        return action if action in self.map.neighbors[state] else state
    
    def action_cost(self, s, action, s1):
        """The distance (cost) to go from s to s1."""
        return self.map.distances[s, s1]
    
    def h(self, node):
        "Straight-line distance between state and the goal."
        locs = self.map.locations
        return straight_line_distance(locs[node.state], locs[self.goal])
    

def straight_line_distance(A, B):
    """Straight-line distance between two points."""
    return sum(abs(a - b)**2 for (a, b) in zip(A, B)) ** 0.5


class Map:
    """A map of places in a 2D world: a graph with vertexes and links between them."""

    def __init__(self, links, locations=None, directed=False):
        if not hasattr(links, 'items'): # Distances are 1 by default
            links = {link: 1 for link in links}
        if not directed:
            for (v1, v2) in list(links):
                links[v2, v1] = links[v1, v2]
        self.distances = links
        self.neighbors = multimap(links.items()) 
        self.locations = locations or defaultdict(lambda: (0, 0))

        
def multimap(pairs) -> dict:
    """Given (key, val) pairs, make a dict of {key: [val,...]}."""
    result = defaultdict(list)
    for key, val in pairs:
        result[key].append(val)
    return result


class GridProblem(Problem):
    """Finding a path on a 2D grid with obstacles."""

    def __init__(self, initial=(15, 30), goal=(130, 30), obstacles=(), **kwds):
        kwds['initial'] = initial
        kwds['goal'] = goal
        kwds['obstacles'] = set(obstacles) - {initial, goal}
        Problem.__init__(self, **kwds)

    directions = [(-1, -1), (0, -1), (1, -1),
                  (-1, 0),           (1,  0),
                  (-1, +1), (0, +1), (1, +1)]
    
    def action_cost(self, s, action, s1): return straight_line_distance(s, s1)
    
    def h(self, node): return straight_line_distance(node.state, self.goal)
                  
    def result(self, state, action): 
        """Both states and actions are represented by (x, y) pairs."""
        return action if action not in self.obstacles else state
    
    def actions(self, state):
        """You can move one cell in any of `directions` to a non-obstacle cell."""
        x, y = state
        possible_moves = set()
        for dx, dy in self.directions:
            possible_moves.add((x + dx, y + dy))
            
        return possible_moves - self.obstacles

# =============================================================================
# 5. BEST FIRST SEARCH AND ITS DERIVATIVES
# =============================================================================

def best_first_search(problem, f):
    """Search nodes with minimum f(node) value first."""
    # The use of the global variable 'reached' is preserved as specified in the code.
    global reached 
    node = Node(problem.initial)
    frontier = PriorityQueue([node], key=f)
    reached = {problem.initial: node}
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            return node
        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
    return failure

def g(n): return n.path_cost

def astar_search(problem, h=None):
    """Search nodes with minimum f(n) = g(n) + h(n)."""
    h = h or problem.h
    return best_first_search(problem, f=lambda n: g(n) + h(n))


def uniform_cost_search(problem):
    """Search nodes with minimum path cost first."""
    return best_first_search(problem, f=g)

# =============================================================================
# 6. AUXILIARY VISUALIZATION FUNCTIONS 
# =============================================================================

def plot_grid_problem(grid, solution, reached=(), title='Search', show=True):
    """Use matplotlib to plot the grid, obstacles, solution, and reached."""
    reached = list(reached)
    plt.figure(figsize=(16, 10))
    plt.axis('off'); plt.axis('equal')
    plt.scatter(*transpose(grid.obstacles), marker='s', color='darkgrey')
    plt.scatter(*transpose(reached), 1**2, marker='.', c='blue')
    plt.scatter(*transpose(path_states(solution)), marker='s', c='blue')
    plt.scatter(*transpose([grid.initial]), 9**2, marker='D', c='green')
    plt.scatter(*transpose([grid.goal]), 9**2, marker='8', c='red')
    if show: plt.show()
    print('{} {} search: {:.1f} path cost, {:,d} states reached'
          .format(' ' * 10, title, solution.path_cost, len(reached)))
    
def plots(grid):
    """Plot the results of the primary search algorithms (A* and UCS) for this grid."""
    solution_astar = astar_search(grid)
    plot_grid_problem(grid, solution_astar, reached, 'A* search')
    
    solution_ucs = uniform_cost_search(grid)
    plot_grid_problem(grid, solution_ucs, reached, 'Uniform Cost Search')
    
def transpose(matrix): return list(zip(*matrix))

# =============================================================================
# 7. ROMANIA MAP EXAMPLE (Test data)
# =============================================================================

romania = Map(
    {('O', 'Z'):  71, ('O', 'S'): 151, ('A', 'Z'): 75, ('A', 'S'): 140, ('A', 'T'): 118, 
     ('L', 'T'): 111, ('L', 'M'):  70, ('D', 'M'): 75, ('C', 'D'): 120, ('C', 'R'): 146, 
     ('C', 'P'): 138, ('R', 'S'):  80, ('F', 'S'): 99, ('B', 'F'): 211, ('B', 'P'): 101, 
     ('B', 'G'):  90, ('B', 'U'):  85, ('H', 'U'): 98, ('E', 'H'):  86, ('U', 'V'): 142, 
     ('I', 'V'):  92, ('I', 'N'):  87, ('P', 'R'): 97},
    {'A': ( 76, 497), 'B': (400, 327), 'C': (246, 285), 'D': (160, 296), 'E': (558, 294), 
     'F': (285, 460), 'G': (368, 257), 'H': (548, 355), 'I': (488, 535), 'L': (162, 379),
     'M': (160, 343), 'N': (407, 561), 'O': (117, 580), 'P': (311, 372), 'R': (227, 412),
     'S': (187, 463), 'T': ( 83, 414), 'U': (471, 363), 'V': (535, 473), 'Z': (92, 539)})

# Test examples (not important for Reverse Engineering, just part of the code)
r0 = RouteProblem('A', 'A', map=romania)
r1 = RouteProblem('A', 'B', map=romania)
r2 = RouteProblem('N', 'L', map=romania)
r3 = RouteProblem('E', 'T', map=romania)
r4 = RouteProblem('O', 'M', map=romania)