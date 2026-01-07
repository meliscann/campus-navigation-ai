from mdp.mdp import MDP

class CampusMDP(MDP):
    """
    Campus-specific MDP.

    State: (floor, x, y)
    Actions:
        - NOOP
        - 4-neighborhood moves: UP/DOWN/LEFT/RIGHT
        - Vertical moves depending on cell type: E (elevator), S (stairs), X (escalator)

    Stochasticity:
        - crowd_stay_prob: probability of staying in place on horizontal moves
        - slip_prob: probability of slipping to a side cell on horizontal moves
        - elevator_delay_prob: probability elevator action fails (stay)
        - escalator_fail_prob: probability escalator action fails (stay)

    Rewards:
        - Transition-based reward (R_transition) uses step_cost / elevator_cost / etc.
        - Reaching terminal gets goal_reward (once, via transition reward).
    """

    def __init__(
        self,
        campus_map,
        terminals,
        gamma=0.95,
        step_cost=1.0,
        elevator_cost=2.0,
        stair_cost=3.0,
        escalator_cost=1.0,
        goal_reward=100.0,
        crowd_stay_prob=0.05,
        slip_prob=0.10,
        elevator_delay_prob=0.30,
        escalator_fail_prob=0.15,
        default_state_reward=-0.1,
    ):

        # Basic config
        self.campus_map = campus_map
        self.terminals = set(terminals)
        self.gamma = float(gamma)

        self.step_cost = float(step_cost)
        self.elevator_cost = float(elevator_cost)
        self.stair_cost = float(stair_cost)
        self.escalator_cost = float(escalator_cost)
        self.goal_reward = float(goal_reward)

        self.crowd_stay_prob = float(crowd_stay_prob)
        self.slip_prob = float(slip_prob)
        self.elevator_delay_prob = float(elevator_delay_prob)
        self.escalator_fail_prob = float(escalator_fail_prob)
        self.default_state_reward = float(default_state_reward)

        # Validate probabilities
        if not (0.0 <= self.crowd_stay_prob <= 1.0):
            raise ValueError("crowd_stay_prob must be in [0,1]")
        if not (0.0 <= self.slip_prob <= 1.0):
            raise ValueError("slip_prob must be in [0,1]")
        if self.crowd_stay_prob + self.slip_prob > 1.0:
            raise ValueError("crowd_stay_prob + slip_prob must be <= 1.0")
        if not (0.0 <= self.elevator_delay_prob <= 1.0):
            raise ValueError("elevator_delay_prob must be in [0,1]")
        if not (0.0 <= self.escalator_fail_prob <= 1.0):
            raise ValueError("escalator_fail_prob must be in [0,1]")

        # Dimensions
        self.max_floors = len(campus_map)
        self.max_y = len(campus_map[0])
        self.max_x = len(campus_map[0][0])

        # States
        states = set()
        for f in range(self.max_floors):
            for y in range(self.max_y):
                for x in range(self.max_x):
                    if campus_map[f][y][x] != "#":
                        states.add((f, x, y))

        # Action list
        actlist = [
            "NOOP",
            "UP", "DOWN", "LEFT", "RIGHT",
            "ELEVATOR_UP", "ELEVATOR_DOWN",
            "STAIR_UP", "STAIR_DOWN",
            "ESCALATOR_UP", "ESCALATOR_DOWN",
        ]

        reward = {s: self.default_state_reward for s in states}
        for t in self.terminals:
            if t in reward:
                reward[t] = self.goal_reward

        super().__init__(
            init=None,
            actlist=actlist,
            terminals=self.terminals,
            transitions=None,
            reward=reward,
            states=states,
            gamma=self.gamma
        )


    # Helpers
    def in_bounds(self, f, x, y):
        return 0 <= f < self.max_floors and 0 <= x < self.max_x and 0 <= y < self.max_y

    def is_wall(self, f, x, y):
        return self.campus_map[f][y][x] == "#"

    def cell(self, s):
        f, x, y = s
        return self.campus_map[f][y][x]

    def move(self, s, dx, dy):
        f, x, y = s
        nx, ny = x + dx, y + dy
        if not self.in_bounds(f, nx, ny) or self.is_wall(f, nx, ny):
            return s
        return (f, nx, ny)

    def vertical(self, s, df):
        f, x, y = s
        nf = f + df
        if not self.in_bounds(nf, x, y) or self.is_wall(nf, x, y):
            return s
        return (nf, x, y)

 
    # MDP interface
    def actions(self, s):
        """Available actions from a state."""
        if s in self.terminals:
            return ["NOOP"]

        f, x, y = s
        acts = []

        # Horizontal moves
        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }
        for a, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if self.in_bounds(f, nx, ny) and not self.is_wall(f, nx, ny):
                acts.append(a)

        # Vertical moves based on special cells
        c = self.cell(s)
        if c == "E":
            if f < self.max_floors - 1 and not self.is_wall(f + 1, x, y):
                acts.append("ELEVATOR_UP")
            if f > 0 and not self.is_wall(f - 1, x, y):
                acts.append("ELEVATOR_DOWN")
        elif c == "S":
            if f < self.max_floors - 1 and not self.is_wall(f + 1, x, y):
                acts.append("STAIR_UP")
            if f > 0 and not self.is_wall(f - 1, x, y):
                acts.append("STAIR_DOWN")
        elif c == "X":
            if f < self.max_floors - 1 and not self.is_wall(f + 1, x, y):
                acts.append("ESCALATOR_UP")
            if f > 0 and not self.is_wall(f - 1, x, y):
                acts.append("ESCALATOR_DOWN")

        return acts if acts else ["NOOP"]

    def R(self, s):
        """State reward fallback."""
        return self.reward.get(s, self.default_state_reward)

    def R_transition(self, s, a, s2):
        """Transition-based reward: costs + terminal bonus."""
        # If transition reaches terminal, give goal reward
        if s2 in self.terminals:
            return 0.0

        # Penalize actions
        if a in ("UP", "DOWN", "LEFT", "RIGHT"):
            return -self.step_cost
        if "ELEVATOR" in a:
            return -self.elevator_cost
        if "STAIR" in a:
            return -self.stair_cost
        if "ESCALATOR" in a:
            return -self.escalator_cost

        # NOOP or unknown
        return self.default_state_reward

    def T(self, s, a):
        """Transition model T(s,a) -> [(p, s')]. Probabilities sum to 1."""
        if s in self.terminals or a == "NOOP":
            return [(1.0, s)]

        # Horizontal actions with slip + stay
        if a in ("UP", "DOWN", "LEFT", "RIGHT"):
            dirs = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
            dx, dy = dirs[a]
            intended = self.move(s, dx, dy)

            # Slip sideways
            if a in ("UP", "DOWN"):
                side1, side2 = self.move(s, -1, 0), self.move(s, 1, 0)
            else:
                side1, side2 = self.move(s, 0, -1), self.move(s, 0, 1)

            p_stay = self.crowd_stay_prob
            p_slip_total = self.slip_prob
            p_intended = 1.0 - p_stay - p_slip_total
            p_side = p_slip_total / 2.0

            probs = {}
            for p, sp in [
                (p_intended, intended),
                (p_side, side1),
                (p_side, side2),
                (p_stay, s),
            ]:
                probs[sp] = probs.get(sp, 0.0) + p

            # normalize just in case of float issues
            total = sum(probs.values())
            if total <= 0:
                return [(1.0, s)]
            return [(p / total, sp) for sp, p in probs.items()]

        # Elevator / Escalator have fail probability (stay)
        if "ELEVATOR" in a or "ESCALATOR" in a:
            p_fail = self.elevator_delay_prob if "ELEVATOR" in a else self.escalator_fail_prob
            df = 1 if "UP" in a else -1
            target = self.vertical(s, df)

            # If vertical move is invalid, we must stay
            if target == s:
                return [(1.0, s)]

            return [(1.0 - p_fail, target), (p_fail, s)]

        # Stairs are deterministic
        if "STAIR" in a:
            df = 1 if "UP" in a else -1
            target = self.vertical(s, df)
            return [(1.0, target)]

        return [(1.0, s)]
