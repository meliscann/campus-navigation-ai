from mdp.mdp import MDP

class CampusMDP(MDP):
    def __init__(
        self,
        campus_map,
        terminals,
        gamma=0.95,
        step_cost=1,
        elevator_cost=2,
        stair_cost=3,
        escalator_cost=1,
        goal_reward=100,
        crowd_stay_prob=0.05,
        slip_prob=0.10,
        elevator_delay_prob=0.30,
        escalator_fail_prob=0.15
    ):
        self.campus_map = campus_map
        self.terminals = set(terminals)
        self.gamma = gamma

        self.step_cost = step_cost
        self.elevator_cost = elevator_cost
        self.stair_cost = stair_cost
        self.escalator_cost = escalator_cost
        self.goal_reward = goal_reward

        self.crowd_stay_prob = crowd_stay_prob
        self.slip_prob = slip_prob
        self.elevator_delay_prob = elevator_delay_prob
        self.escalator_fail_prob = escalator_fail_prob

        self.max_floors = len(campus_map)
        self.max_y = len(campus_map[0])
        self.max_x = len(campus_map[0][0])

        states = set()
        for f in range(self.max_floors):
            for y in range(self.max_y):
                for x in range(self.max_x):
                    if campus_map[f][y][x] != "#":
                        states.add((f, x, y))

        actlist = [
            "NOOP", "UP", "DOWN", "LEFT", "RIGHT",
            "ELEVATOR_UP", "ELEVATOR_DOWN",
            "STAIR_UP", "STAIR_DOWN",
            "ESCALATOR_UP", "ESCALATOR_DOWN"
        ]

        reward = {s: (self.goal_reward if s in self.terminals else -0.1) for s in states}

        try:
            super().__init__(init=None, actlist=actlist, terminals=self.terminals,
                            transitions=None, reward=reward, gamma=self.gamma)
        except TypeError:
            super().__init__(init=None, actlist=actlist, terminals=self.terminals, gamma=self.gamma)
            self.reward = reward

        self.states = states

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

    def actions(self, s):
        if s in self.terminals:
            return ["NOOP"]

        f, x, y = s
        acts = []

        directions = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
        for a, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if self.in_bounds(f, nx, ny) and not self.is_wall(f, nx, ny):
                acts.append(a)

        c = self.cell(s)
        if c == "E":
            if f < self.max_floors - 1 and not self.is_wall(f + 1, x, y): acts.append("ELEVATOR_UP")
            if f > 0 and not self.is_wall(f - 1, x, y): acts.append("ELEVATOR_DOWN")
        elif c == "S":
            if f < self.max_floors - 1 and not self.is_wall(f + 1, x, y): acts.append("STAIR_UP")
            if f > 0 and not self.is_wall(f - 1, x, y): acts.append("STAIR_DOWN")
        elif c == "X":
            if f < self.max_floors - 1 and not self.is_wall(f + 1, x, y): acts.append("ESCALATOR_UP")
            if f > 0 and not self.is_wall(f - 1, x, y): acts.append("ESCALATOR_DOWN")

        return acts if acts else ["NOOP"]

    def R(self, s):
        return self.reward.get(s, -0.1)

    def T(self, s, a):
        if s in self.terminals or a == "NOOP":
            return [(1.0, s)]

        if a in ("UP", "DOWN", "LEFT", "RIGHT"):
            dirs = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
            dx, dy = dirs[a]
            intended = self.move(s, dx, dy)

            if a in ("UP", "DOWN"):
                side1, side2 = self.move(s, -1, 0), self.move(s, 1, 0)
            else:
                side1, side2 = self.move(s, 0, -1), self.move(s, 0, 1)

            p_stay = self.crowd_stay_prob
            p_slip_total = self.slip_prob
            p_intended = 1.0 - p_stay - p_slip_total
            p_side = p_slip_total / 2.0

            probs = {}
            for p, sp in [(p_intended, intended), (p_side, side1), (p_side, side2), (p_stay, s)]:
                probs[sp] = probs.get(sp, 0) + p
            return [(p, sp) for sp, p in probs.items()]

        if "ELEVATOR" in a or "ESCALATOR" in a:
            p_fail = self.elevator_delay_prob if "ELEVATOR" in a else self.escalator_fail_prob
            df = 1 if "UP" in a else -1
            target = self.vertical(s, df)
            if target == s: return [(1.0, s)]
            return [(1.0 - p_fail, target), (p_fail, s)]

        if "STAIR" in a:
            df = 1 if "UP" in a else -1
            return [(1.0, self.vertical(s, df))]

        return [(1.0, s)]