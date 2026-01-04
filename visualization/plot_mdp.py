import math
import matplotlib.pyplot as plt

WALL = "#"

# Aynı renk mapping'ini kullan (senin mevcut _cell_color'un)
def _cell_color(cell: str) -> float:
    mapping = {
        "#": 0.00,  # wall
        ".": 1.00,  # free

        # vertical transfer cells
        "E": 0.75,  # elevator
        "S": 0.60,  # stairs
        "X": 0.45,  # escalator

        # semantic walkable areas
        "R": 0.88,  # restaurant
        "O": 0.70,  # student affairs
        "L": 0.82,  # library
        "C": 0.92,  # classroom
        "F": 0.78,  # faculty
        "T": 0.96,  # terrace
    }
    return mapping.get(cell, 1.0)


def _is_walkable(cell: str) -> bool:
    return cell != WALL


def _utility_floor_matrix(campus_map, U: dict, f: int):
    """Return 2D matrix (rows x cols) of utility values for floor f, NaN on walls."""
    floor = campus_map[f]
    rows, cols = len(floor), len(floor[0])

    mat = [[float("nan")] * cols for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            cell = floor[y][x]
            if _is_walkable(cell):
                mat[y][x] = U.get((f, x, y), float("nan"))
    return mat


def _normalize_matrix(mat):
    """Normalize numeric matrix to 0..1 ignoring NaNs."""
    vals = []
    for row in mat:
        for v in row:
            if not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
    if not vals:
        return mat  # nothing to normalize

    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        # all same
        return [[0.5 if not (isinstance(v, float) and math.isnan(v)) else float("nan") for v in row] for row in mat]

    out = []
    for row in mat:
        new_row = []
        for v in row:
            if isinstance(v, float) and math.isnan(v):
                new_row.append(float("nan"))
            else:
                new_row.append((v - vmin) / (vmax - vmin))
        out.append(new_row)
    return out


def plot_mdp_floor(campus_map, U: dict, pi: dict, f: int, title: str = ""):
    """
    1 fig içinde:
    - sol: utility heatmap (normalize)
    - sağ: policy arrows (walkable cells)
    """
    floor = campus_map[f]
    rows, cols = len(floor), len(floor[0])

    # --- Base map for background (semantic map) ---
    base = [[_cell_color(floor[y][x]) for x in range(cols)] for y in range(rows)]

    # --- Utility heatmap (normalized) ---
    Umat = _utility_floor_matrix(campus_map, U, f)
    Unorm = _normalize_matrix(Umat)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ========== LEFT: Utility ==========
    ax = axes[0]
    ax.imshow(base, interpolation="nearest", origin="upper")
    ax.imshow(Unorm, interpolation="nearest", origin="upper", alpha=0.55)  # utility overlay

    ax.set_title(f"Utility (Floor {f})")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, linewidth=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ========== RIGHT: Policy ==========
    ax2 = axes[1]
    ax2.imshow(base, interpolation="nearest", origin="upper")
    ax2.set_title(f"Policy (Floor {f})")
    ax2.set_xticks(range(cols))
    ax2.set_yticks(range(rows))
    ax2.grid(True, linewidth=0.3)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    # Arrow mapping (x,y) coordinate system in imshow:
    # x increases right, y increases down.
    action_to_vec = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    # Draw arrows / symbols
    for y in range(rows):
        for x in range(cols):
            cell = floor[y][x]
            if not _is_walkable(cell):
                continue

            a = pi.get((f, x, y), None)
            if a is None:
                continue

            # Horizontal movement arrows
            if a in action_to_vec:
                dx, dy = action_to_vec[a]
                # oklar küçük görünsün
                ax2.arrow(
                    x, y,
                    0.35 * dx, 0.35 * dy,
                    head_width=0.18,
                    head_length=0.18,
                    length_includes_head=True,
                    linewidth=0.8
                )

            # Vertical actions: sembol yaz
            elif a in ("ELEVATOR_UP", "ELEVATOR_DOWN", "STAIR_UP", "STAIR_DOWN", "ESCALATOR_UP", "ESCALATOR_DOWN"):
                if "UP" in a:
                    sym = "↑"
                else:
                    sym = "↓"

                # E/S/X harfini de koy
                # ELEVATOR -> E, STAIR -> S, ESCALATOR -> X
                if a.startswith("ELEVATOR"):
                    kind = "E"
                elif a.startswith("STAIR"):
                    kind = "S"
                else:
                    kind = "X"

                ax2.text(x, y, f"{kind}{sym}", ha="center", va="center", fontsize=8)

            elif a == "NOOP":
                ax2.text(x, y, "•", ha="center", va="center", fontsize=10)

    fig.suptitle(title if title else f"MDP Visualization - Floor {f}", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_mdp_all_floors(campus_map, U: dict, pi: dict, floors=None, title_prefix="MDP"):
    """
    Çok fazla fig açmamak için her floor için ayrı fig değil,
    seçtiğin floor'lar için sırayla tek tek çiz.
    """
    if floors is None:
        floors = list(range(len(campus_map)))

    for f in floors:
        plot_mdp_floor(campus_map, U, pi, f, title=f"{title_prefix} (Floor {f})")
