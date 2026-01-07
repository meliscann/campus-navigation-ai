"""
Shared utility functions used by the AIMA-based Search and MDP modules.

Includes:
- PriorityQueue (stable tie-breaking; safe for A*/UCS when priorities are equal)
- Basic iterable helpers
- Vector/grid helpers (orientations, turns) used by MDP
- Memoization helper
"""

import collections
import collections.abc
import functools
import heapq
from itertools import chain
import numpy as np

# Priority Queue
class PriorityQueue:
    """
    A queue in which the minimum (or maximum) element (as determined by f and order)
    is returned first.

    Fixes a common heapq pitfall:
      If two items have the same priority, Python may try to compare the items
      themselves (which can raise TypeError for custom objects).
    We solve it by adding a monotonic counter as a tiebreaker.

    - order='min': pops smallest f(item)
    - order='max': pops largest f(item)
    Supports dict-like lookup by item equality.
    """

    def __init__(self, order="min", f=lambda x: x):
        self.heap = []
        self._counter = 0

        if order == "min":
            self.f = f
        elif order == "max":
            self.f = lambda x: -f(x)
        else:
            raise ValueError("Order must be either 'min' or 'max'.")

    def append(self, item):
        """Insert item with its priority into the heap."""
        self._counter += 1
        heapq.heappush(self.heap, (self.f(item), self._counter, item))

    def extend(self, items):
        """Insert each item in items."""
        for item in items:
            self.append(item)

    def pop(self):
        """Pop and return the item with min/max priority (depending on order)."""
        if not self.heap:
            raise Exception("Trying to pop from empty PriorityQueue.")
        return heapq.heappop(self.heap)[2]

    def __len__(self):
        """Return the number of items in the queue."""
        return len(self.heap)

    def __contains__(self, key):
        """Return True if key is in the queue."""
        return any(item == key for _, _, item in self.heap)

    def __getitem__(self, key):
        """
        Return the stored priority value associated with key.
        Raises KeyError if key is not present.
        """
        for priority, _, item in self.heap:
            if item == key:
                return priority
        raise KeyError(f"{key} is not in the priority queue")

    def __delitem__(self, key):
        """Delete the first occurrence of key."""
        for i, (_, _, item) in enumerate(self.heap):
            if item == key:
                del self.heap[i]
                heapq.heapify(self.heap)
                return
        raise KeyError(f"{key} is not in the priority queue")


# Functions on Sequences and Iterables
def sequence(iterable):
    """Convert iterable to a sequence if it is not already one."""
    return iterable if isinstance(iterable, collections.abc.Sequence) else tuple([iterable])

def product(numbers):
    """Return the product of numbers. Example: product([2, 3, 10]) == 60"""
    result = 1
    for x in numbers:
        result *= x
    return result

def first(iterable, default=None):
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)

def flatten(seqs):
    """Flatten a list (or iterable) of iterables into a single list."""
    return list(chain.from_iterable(seqs))


# Mathematical and Statistical util functions
def vector_add(a, b):
    """Component-wise addition of two vectors."""
    if not (a and b):
        return a or b
    if hasattr(a, "__iter__") and hasattr(b, "__iter__"):
        assert len(a) == len(b)
        return list(map(vector_add, a, b))
    try:
        return a + b
    except TypeError as e:
        raise Exception("Inputs must be of compatible size/type for addition!") from e

def euclidean_distance(x, y):
    """Euclidean distance between two vectors."""
    return float(np.sqrt(sum((_x - _y) ** 2 for _x, _y in zip(x, y))))

def manhattan_distance(x, y):
    """Manhattan distance between two vectors."""
    return sum(abs(_x - _y) for _x, _y in zip(x, y))


# Grid / Heading helpers (used by MDP)
orientations = EAST, NORTH, WEST, SOUTH = [(1, 0), (0, 1), (-1, 0), (0, -1)]
turns = LEFT, RIGHT = (+1, -1)

def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]

def turn_right(heading):
    return turn_heading(heading, RIGHT)

def turn_left(heading):
    return turn_heading(heading, LEFT)


# Memoization
def memoize(fn, slot=None, maxsize=32):
    """
    Memoize fn: remember computed values.

    If slot is specified, store result in that slot of first argument.
    If slot is falsy, use functools.lru_cache for caching.
    """
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            val = fn(obj, *args)
            setattr(obj, slot, val)
            return val
    else:
        @functools.lru_cache(maxsize=maxsize)
        def memoized_fn(*args):
            return fn(*args)

    return memoized_fn
