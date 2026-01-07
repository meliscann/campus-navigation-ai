import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WALL = "#"

def _cell_color(cell: str) -> float:
    """
    Map cells to grayscale values for imshow.
    0.0 = black, 1.0 = white
    """
    mapping = {
        "#": 0.00,  # wall
        ".": 1.00,  # free

        # vertical transfer cells
        "E": 0.75,  # elevator
        "S": 0.60,  # stairs
        "X": 0.45,  # escalator

        # semantic walkable areas
        "R": 0.85,  # restaurant
        "A": 0.70,  # student affairs
        "L": 0.82,  # library
        "C": 0.90,  # classroom
        "F": 0.75,  # faculty
        "T": 0.40,  # terrace
    }
    return mapping.get(cell, 0.95)  # unknown but walkable

def _overlay_labels(ax, floor, show_labels=True):
    """Write letters on top of special cells."""
    if not show_labels:
        return
    rows = len(floor)
    cols = len(floor[0])

    for y in range(rows):
        for x in range(cols):
            cell = floor[y][x]
            if cell in ("#", "."):
                continue
            ax.text(x, y, cell, ha="center", va="center", fontsize=8, fontweight="bold")

def _draw_floor(ax, floor, title="", show_labels=True):
    rows = len(floor)
    cols = len(floor[0])

    img = [[_cell_color(floor[y][x]) for x in range(cols)] for y in range(rows)]
    ax.imshow(img, interpolation="nearest", origin="upper")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True, linewidth=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    _overlay_labels(ax, floor, show_labels=show_labels)
    ax.set_title(title)

def _add_legend(fig):
    """Figure-level legend explaining markers (consistent with grayscale map)."""
    legend_items = [
        Line2D([0], [0], color="black", linewidth=2, marker="o", markersize=4, label="Path"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="white",
               markeredgecolor="black", markersize=9, label="Start"),
        Line2D([0], [0], marker="*", linestyle="None", markerfacecolor="white",
               markeredgecolor="black", markersize=12, label="Goal"),
        Line2D([], [], linestyle="None", label="Cells: E=elevator, S=stairs, X=escalator"),
        Line2D([], [], linestyle="None", label="Areas: R=restaurant, A=student affairs, L=library, C=classroom, F=faculty, T=terrace"),
    ]

    fig.legend(
        handles=legend_items,
        loc="upper center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.96)
    )

def plot_floor_with_path(campus_map, path, floor_index: int, title: str = ""):
    """Draw a single floor and overlay the part of the path on this floor."""
    floor = campus_map[floor_index]
    fig, ax = plt.subplots(figsize=(6, 4))

    _draw_floor(ax, floor, title=title if title else f"Floor {floor_index}", show_labels=True)

    if path:
        pts = [(x, y) for (f, x, y) in path if f == floor_index]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2, marker="o", markersize=3)

            start = path[0]
            goal = path[-1]
            if start[0] == floor_index:
                ax.scatter([start[1]], [start[2]], marker="s", s=80)
                ax.text(start[1], start[2], "Start", fontsize=9, va="bottom")
            if goal[0] == floor_index:
                ax.scatter([goal[1]], [goal[2]], marker="*", s=120)
                ax.text(goal[1], goal[2], "Goal", fontsize=9, va="bottom")

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

def plot_path_all_floors(campus_map, path, title_prefix: str = "Path"):
    """Plot all floors that appear in the path in ONE figure using subplots."""
    if not path:
        print("No path to plot.")
        return

    floors_in_path = sorted({f for (f, _, _) in path})
    n = len(floors_in_path)

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, f in enumerate(floors_in_path):
        ax = axes[idx]
        floor = campus_map[f]

        _draw_floor(ax, floor, title=f"{title_prefix} (Floor {f})", show_labels=True)

        pts = [(x, y) for (ff, x, y) in path if ff == f]
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, linewidth=2, marker="o", markersize=3)

            start = path[0]
            goal = path[-1]
            if start[0] == f:
                ax.scatter([start[1]], [start[2]], marker="s", s=80)
                ax.text(start[1], start[2], "Start", fontsize=9, va="bottom")
            if goal[0] == f:
                ax.scatter([goal[1]], [goal[2]], marker="*", s=120)
                ax.text(goal[1], goal[2], "Goal", fontsize=9, va="bottom")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{title_prefix} - All Floors", fontsize=14)

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()
