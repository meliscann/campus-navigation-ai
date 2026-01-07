# Per-action time and effort components

TIME_COST = {
    "MOVE": 2,

    "ELEVATOR_UP": 3,
    "ELEVATOR_DOWN": 3,

    "ESCALATOR_UP": 2,
    "ESCALATOR_DOWN": 2,

    "STAIR_UP": 2,
    "STAIR_DOWN": 1,

    "NOOP": 0,
}

EFFORT_COST = {
    "MOVE": 1,

    "ELEVATOR_UP": 1,
    "ELEVATOR_DOWN": 1,

    "ESCALATOR_UP": 2,
    "ESCALATOR_DOWN": 2,

    "STAIR_UP": 4,
    "STAIR_DOWN": 2,

    "NOOP": 0,
}


def action_cost(
    action: str,
    mode: str = "total",
    alpha: float = 1.0,
    beta: float = 1.0
) -> float:
    """
    mode:
      - 'time'   : minimize time only
      - 'effort' : minimize effort only
      - 'total'  : weighted combination (default)
    """
    t = TIME_COST.get(action, 2)
    e = EFFORT_COST.get(action, 2)

    if mode == "time":
        return t
    elif mode == "effort":
        return e
    else:
        return alpha * t + beta * e
