"""
MDP Implementation for Campus Navigation Project (Based on AIMA Chapter 16)

Includes:
- MDP base class
- GridMDP example class (optional)
- Utility-based method: Value Iteration
"""

from collections import defaultdict
from core.utils import vector_add, orientations, turn_right, turn_left

#  MDP BASE CLASSES
class MDP:
    """Markov Decision Process representation.

    Parameters:
        init: Initial state
        actlist: List (or other iterable) of actions
        terminals: Iterable of terminal states
        transitions: Transition model T[s][a] -> [(p, s')]
        reward: Reward function mapping R[s] -> float
        states: Set of all states
        gamma: Discount factor in (0, 1]
    """

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("Gamma must satisfy 0 < gamma <= 1")

        self.transitions = transitions or {}
        self.states = states or self.get_states_from_transitions(self.transitions)
        if self.states is None:
            raise ValueError("States could not be inferred. Provide `states` or a valid `transitions` dict.")

        self.init = init
        self.actlist = list(actlist) if actlist is not None else []
        self.terminals = set(terminals) if terminals is not None else set()
        self.gamma = gamma
        self.reward = reward or {s: 0 for s in self.states}

    def R(self, state):
        """Return reward of a state."""
        return self.reward.get(state, 0)

    def T(self, state, action):
        """Return transition model for (state, action)."""
        if not self.transitions:
            raise ValueError("Transition table missing.")
        return self.transitions[state][action]

    def actions(self, state):
        """Return available actions for state."""
        return [None] if state in self.terminals else self.actlist

    @staticmethod
    def get_states_from_transitions(transitions):
        """Extract states from a transitions dict."""
        if not isinstance(transitions, dict) or not transitions:
            return None

        from_keys = set(transitions.keys())
        to_states = set(
            s2
            for actions in transitions.values()
            for effects in actions.values()
            for (p, s2) in effects
        )
        return from_keys.union(to_states)


class GridMDP(MDP):
    """A 2D grid-based MDP (optional example)."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=0.9):
        grid = list(reversed(grid))

        reward, states = {}, set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid

        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x] is not None:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]

        actlist = orientations
        transitions = {s: {a: self.calculate_T(s, a) for a in actlist} for s in states}

        super().__init__(init=init, actlist=actlist, terminals=terminals,
                         transitions=transitions, reward=reward, states=states, gamma=gamma)

    def calculate_T(self, state, action):
        if action is None:
            return [(1.0, state)]

        return [
            (0.8, self.go(state, action)),
            (0.1, self.go(state, turn_right(action))),
            (0.1, self.go(state, turn_left(action))),
        ]

    def go(self, state, direction):
        new_state = tuple(vector_add(state, direction))
        return new_state if new_state in self.states else state

    def to_grid(self, mapping):
        return list(reversed([
            [mapping.get((x, y), None) for x in range(self.cols)]
            for y in range(self.rows)
        ]))

    def to_arrows(self, policy):
        chars = {(1, 0): '>', (0, 1): '^', (-1, 0): '<', (0, -1): 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


#  MDP ALGORITHMS
def q_value(mdp, s, a, U):
    """Compute Q(s,a) = Σ p(s'|s,a) [ R + γ U(s') ]."""
    if a is None:
        return mdp.R(s)

    total = 0.0
    for p, s2 in mdp.T(s, a):
        if hasattr(mdp, "R_transition"):
            r = mdp.R_transition(s, a, s2)
        else:
            r = mdp.R(s)

        total += p * (r + mdp.gamma * U.get(s2, 0))

    return total


def value_iteration(mdp, epsilon=0.001):
    """Value Iteration algorithm.

    Returns:
        U: dict mapping state -> utility
    """
    U = {s: 0.0 for s in mdp.states}
    gamma = mdp.gamma

    if gamma == 1:
        raise ValueError("gamma=1 can prevent convergence in continuing tasks; use gamma < 1.")

    while True:
        U_prev = U.copy()
        delta = 0.0

        for s in mdp.states:
            if s in mdp.terminals:
                U[s] = mdp.R(s)
                continue

            U[s] = max(q_value(mdp, s, a, U_prev) for a in mdp.actions(s))
            delta = max(delta, abs(U[s] - U_prev[s]))

        if delta <= epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    """Return best policy π* for utility mapping U."""
    pi = {}
    for s in mdp.states:
        if s in mdp.terminals:
            pi[s] = None
        else:
            pi[s] = max(mdp.actions(s), key=lambda a: q_value(mdp, s, a, U))
    return pi


def expected_utility(a, s, U, mdp):
    """EU(s,a) = Σ p(s'|s,a) U(s')."""
    return sum(p * U[s1] for p, s1 in mdp.T(s, a))
