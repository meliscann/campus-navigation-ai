from search.search import Problem

class CampusProblem(Problem):
    def __init__(self, initial, goal, campus_map):
        super().__init__(initial, goal)
        self.campus_map = campus_map

    def actions(self, state):
        floor, x, y = state
        actions = []

        max_floors = len(self.campus_map)
        max_y = len(self.campus_map[floor])
        max_x = len(self.campus_map[floor][0])

        # 4-directional movements
        directions = {
            "UP": (0, -1),
            "DOWN": (0, 1),
            "LEFT": (-1, 0),
            "RIGHT": (1, 0)
        }

        for action, (dx, dy) in directions.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < max_x and 0 <= ny < max_y:
                if self.campus_map[floor][ny][nx] != "#":
                    actions.append(action)

        # vertical movements
        cell = self.campus_map[floor][y][x]

        # Elevator
        if cell == "E":
            if floor < max_floors - 1:
                actions.append("ELEVATOR_UP")
            if floor > 0:
                actions.append("ELEVATOR_DOWN")

        # Staircase
        if cell == "S":
            if floor < max_floors - 1:
                actions.append("STAIR_UP")
            if floor > 0:
                actions.append("STAIR_DOWN")

        # Escalator
        if cell == "X":
            if floor < max_floors - 1:
                actions.append("ESCALATOR_UP")
            if floor > 0:
                actions.append("ESCALATOR_DOWN")

        return actions


    def result(self, state, action):
        floor, x, y = state

        # 4-directional moves
        if action == "UP":
            return (floor, x, y - 1)
        if action == "DOWN":
            return (floor, x, y + 1)
        if action == "LEFT":
            return (floor, x - 1, y)
        if action == "RIGHT":
            return (floor, x + 1, y)

        # Vertical moves (same x,y; only floor changes)
        if action == "ELEVATOR_UP":
            return (floor + 1, x, y)
        if action == "ELEVATOR_DOWN":
            return (floor - 1, x, y)

        if action == "STAIR_UP":
            return (floor + 1, x, y)
        if action == "STAIR_DOWN":
            return (floor - 1, x, y)

        if action == "ESCALATOR_UP":
            return (floor + 1, x, y)
        if action == "ESCALATOR_DOWN":
            return (floor - 1, x, y)

        return state


    def path_cost(self, c, state1, action, state2):
        # Base costs
        if action in ("UP", "DOWN", "LEFT", "RIGHT"):
            step_cost = 1

        elif action in ("ESCALATOR_UP", "ESCALATOR_DOWN"):
            step_cost = 4

        elif action in ("ELEVATOR_UP", "ELEVATOR_DOWN"):
            step_cost = 2

        elif action in ("STAIR_UP", "STAIR_DOWN"):
            step_cost = 10

        else:
            # unknown action: penalize a bit to discourage it
            step_cost = 5

        return c + step_cost

    def action_cost(self, state1, action, state2):
        return self.path_cost(0, state1, action, state2)

    def goal_test(self, state):
        return state == self.goal
