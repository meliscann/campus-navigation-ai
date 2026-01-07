from typing import List, Tuple, Dict

State = Tuple[int, int, int]  # (floor, x, y)

CATEGORY_TO_CELL: Dict[str, str] = {
    "restaurant": "R",
    "classroom": "C",
    "library": "L",
    "faculty": "F",
    "terrace": "T",
    "student_affairs": "A",
}

def find_cells(campus_map, cell_char: str) -> List[State]:
    goals: List[State] = []
    for f, floor in enumerate(campus_map):
        for y, row in enumerate(floor):
            for x, cell in enumerate(row):
                if cell == cell_char:
                    goals.append((f, x, y))
    return goals
