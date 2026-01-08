# Campus Navigation AI

An intelligent agent-based navigation system for a multi-story university campus. This project implements and compares classical search algorithms (**BFS**, **UCS**, **A***) to solve the navigation problem in a complex 3D grid environment.

## 🚀 Features

*   **Multi-Floor Environment:** A realistic 6-floor campus map with elevators, stairs, and escalators.
*   **Semantic Navigation:** Agents navigate to categories (e.g., *"Find the nearest Restaurant"*) rather than just coordinates.
*   **Dual-Objective Cost Model:** Balances **Time** vs **Physical Effort**.
    *   *Stairs:* Low Time, High Effort (Cost: 12).
    *   *Elevator:* High Time (Wait), Low Effort (Cost: 2).
    *   *Escalator:* Balanced (Cost: 4).
*   **Vizualization:** Real-time Matplotlib visualization of the agent's path across floors.

## 📂 Project Structure

```text
campus-nav-ai/
├── environment/           # Knowledge Base
│   ├── campus_map.py      # Map topology (6 Floors)
│   ├── campus_search.py   # State transitions & Actions
│   ├── costs.py           # Weighted Cost Function (Time/Effort)
│   └── poi.py             # Semantic Indexing
├── search/                # AI Algorithms
│   ├── astar.py           # A* Search (Optimal & Efficient)
│   ├── ucs.py             # Uniform Cost Search (Optimal)
│   ├── bfs.py             # Breadth-First Search (Sub-optimal)
│   ├── heuristics.py      # Cost-Aware Heuristic Function
│   └── common.py          # Abstract Node/Problem classes (AIMA)
├── main_search.py         # Main Entry Point
├── visualization/         # Plotting Tools
└── requirements.txt       # Dependencies
```

## 🛠️ Setup & Requirements

Since you have the source code folder, you only need to install the required Python libraries.

1.  **Open your terminal** in this project directory (`campus-nav-ai`).
2.  **Install dependencies:**

```bash
pip install -r requirements.txt
```

*(Requires Python 3.8+)*


## 🏃‍♂️ How to Run

Run the main script to start the interactive search agent:

```bash
python main_search.py
```

### Example Scenarios to Try

1.  **Baseline Navigation:**
    *   Start: Default
    *   Target: `library`
    *   *Result:* All algorithms find the path. BFS is shortest in steps, A* is efficient.

2.  **Escalator vs Stairs (The "Effort" Test):**
    *   Edit `main_search.py` -> set `initial = (0, 22, 6)` (Near stairs)
    *   Target: `terrace` (Top floor)
    *   *Result:* BFS will ignorantly climb 5 flights of stairs. UCS and A* will walk further to take the elevator.

## 🧠 Algorithms Implemented

| Algorithm | Completeness | Optimality | Description |
| :--- | :---: | :---: | :--- |
| **BFS** | ✅ Yes | ❌ No | Ignores transition costs. Good for unweighted graphs only. |
| **UCS** | ✅ Yes | ✅ Yes | Explores radially based on cost. Guaranteed optimal but slow. |
| **A* Search** | ✅ Yes | ✅ Yes | Uses `Manhattan + VerticalPenalty` heuristic. Optimal & **~50% faster** than UCS. |
