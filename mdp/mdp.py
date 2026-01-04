"""
MDP Implementation for Campus Navigation Project (Based on AIMA Chapter 16)

Includes:
- MDP class hierarchy (MDP, GridMDP)
- Utility-based method (Value Iteration)

All Policy Iteration and POMDP related algorithms have been removed for project focus.
"""

import random
from collections import defaultdict
import numpy as np

from core.utils import vector_add, orientations, turn_right, turn_left


# =============================================================================
#  MDP BASE CLASSES
# =============================================================================

class MDP:
    """Markov Decision Process representation.

    Parameters:
        init: Initial state
        actlist: List or dict of actions
        terminals: Terminal states
        transitions: Transition model T(s, a) → [(p, s')]
        reward: Reward function R(s)
        states: Set of states
        gamma: Discount factor
    """

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("Gamma must satisfy 0 < gamma <= 1")

        self.states = states or self.get_states_from_transitions(transitions)
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        self.transitions = transitions or {}
        self.gamma = gamma
        self.reward = reward or {s: 0 for s in self.states}

    def R(self, state):
        """Return reward of a state."""
        return self.reward[state]

    def T(self, state, action):
        """Return transition model for (state, action)."""
        if not self.transitions:
            raise ValueError("Transition table missing.")
        return self.transitions[state][action]

    def actions(self, state):
        """Return available actions for state."""
        return [None] if state in self.terminals else self.actlist

    def get_states_from_transitions(self, transitions):
        """Extract states from transition dict."""
        if isinstance(transitions, dict):
            from_keys = set(transitions.keys())
            to_states = set(
                tr[1]
                for actions in transitions.values()
                for effects in actions.values()
                for tr in effects
            )
            return from_keys.union(to_states)
        return None


# -----------------------------------------------------------------------------

class GridMDP(MDP):
    """A 2D grid-based MDP."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse()
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

        super().__init__(init, actlist, terminals, transitions, reward, states, gamma)

    def calculate_T(self, state, action):
        if action is None:
            return [(0.0, state)]
        return [
            (0.8, self.go(state, action)),
            (0.1, self.go(state, turn_right(action))),
            (0.1, self.go(state, turn_left(action)))
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


# =============================================================================
#  MDP ALGORITHMS
# =============================================================================

def q_value(mdp, s, a, U):
    """Compute Q-value for (s, a)."""
    if a is None:
        return mdp.R(s)
    total = 0
    for p, s2 in mdp.T(s, a):
        if hasattr(mdp, "R_transition"):
            r = mdp.R_transition(s, a, s2)
        else:
            r = mdp.R(s)
        total += p * (r + mdp.gamma * U[s2])
    return total


def value_iteration(mdp, epsilon=0.001):
    """Value iteration algorithm."""
    U1 = {s: 0 for s in mdp.states}
    gamma = mdp.gamma

    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = max(q_value(mdp, s, a, U) for a in mdp.actions(s))
            delta = max(delta, abs(U1[s] - U[s]))

        if delta <= epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    """Return best policy π* for utility mapping U."""
    return {s: max(mdp.actions(s), key=lambda a: q_value(mdp, s, a, U))
            for s in mdp.states}


def expected_utility(a, s, U, mdp):
    return sum(p * U[s1] for p, s1 in mdp.T(s, a))
