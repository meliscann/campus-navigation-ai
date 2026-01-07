from search.search import Problem

class CampusProblem(Problem):
    def __init__(self, initial, goal, campus_map):
        super().__init__(initial, goal)
        self.campus_map = campus_map
        self.max_floors = len(campus_map)
        self.max_y = len(campus_map[0])
        self.max_x = len(campus_map[0][0])

    def in_bounds(self, f, x, y):
        return 0 <= f < self.max_floors and 0 <= x < self.max_x and 0 <= y < self.max_y

    def is_wall(self, f, x, y):
        return self.campus_map[f][y][x] == "#"

    def actions(self, state):
        floor, x, y = state
        acts = []

        # 4-directional movements
        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        for a, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if self.in_bounds(floor, nx, ny) and not self.is_wall(floor, nx, ny):
                acts.append(a)

        # vertical movements depend on the current cell
        cell = self.campus_map[floor][y][x]

        def can_go(df):
            nf = floor + df
            return self.in_bounds(nf, x, y) and not self.is_wall(nf, x, y)

        if cell == "E":
            if can_go(+1): acts.append("ELEVATOR_UP")
            if can_go(-1): acts.append("ELEVATOR_DOWN")

        elif cell == "S":
            if can_go(+1): acts.append("STAIR_UP")
            if can_go(-1): acts.append("STAIR_DOWN")

        elif cell == "X":
            if can_go(+1): acts.append("ESCALATOR_UP")
            if can_go(-1): acts.append("ESCALATOR_DOWN")

        return acts

    def result(self, state, action):
        floor, x, y = state

        moves = {
            "UP": (0, -1, 0),
            "DOWN": (0, 1, 0),
            "LEFT": (-1, 0, 0),
            "RIGHT": (1, 0, 0),

            "ELEVATOR_UP": (0, 0, +1),
            "ELEVATOR_DOWN": (0, 0, -1),
            "STAIR_UP": (0, 0, +1),
            "STAIR_DOWN": (0, 0, -1),
            "ESCALATOR_UP": (0, 0, +1),
            "ESCALATOR_DOWN": (0, 0, -1),
        }

        if action not in moves:
            return state

        dx, dy, df = moves[action]
        nf, nx, ny = floor + df, x + dx, y + dy

        if not self.in_bounds(nf, nx, ny) or self.is_wall(nf, nx, ny):
            return state
        return (nf, nx, ny)

    def path_cost(self, c, state1, action, state2):
        if action in ("UP", "DOWN", "LEFT", "RIGHT"):
            step_cost = 1
        elif action in ("ESCALATOR_UP", "ESCALATOR_DOWN"):
            step_cost = 4
        elif action in ("ELEVATOR_UP", "ELEVATOR_DOWN"):
            step_cost = 2
        elif action in ("STAIR_UP", "STAIR_DOWN"):
            step_cost = 10
        else:
            step_cost = 5
        return c + step_cost

    def action_cost(self, state1, action, state2):
        return self.path_cost(0, state1, action, state2)

    def is_goal(self, state):
        return state == self.goal
