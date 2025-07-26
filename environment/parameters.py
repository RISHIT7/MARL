HORIZON = 100
W_R = 1.0
W_D = 1.0

# Grid network parameters
GRID_WIDTH = 10
GRID_HEIGHT = 10
NODES = GRID_WIDTH * GRID_HEIGHT  # 100 nodes in a 10x10 grid

# Arrival nodes (top row)
ARRIVAL_NODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Generate grid edges - each node connects to its neighbors (right and down)
EDGES = []

# Horizontal edges (left-right connections)
for row in range(GRID_HEIGHT):
    for col in range(GRID_WIDTH - 1):
        node1 = row * GRID_WIDTH + col + 1
        node2 = row * GRID_WIDTH + col + 2
        EDGES.append((node1, node2))
        EDGES.append((node2, node1))  # Bidirectional

# Vertical edges (up-down connections)
for row in range(GRID_HEIGHT - 1):
    for col in range(GRID_WIDTH):
        node1 = row * GRID_WIDTH + col + 1
        node2 = (row + 1) * GRID_WIDTH + col + 1
        EDGES.append((node1, node2))
        EDGES.append((node2, node1))  # Bidirectional

# Terminal nodes (bottom row)
TERMINAL_NODES = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]

# Simple arrival distribution - equal probability for each arrival node and time step
ARRIVAL_DIST = {}
total_entries = len(ARRIVAL_NODES) * 10  # 10 nodes * 10 time steps = 100 entries
probability_per_entry = 1.0 / total_entries

for node in ARRIVAL_NODES:
    for time_step in range(1, 11):
        ARRIVAL_DIST[(node, time_step)] = probability_per_entry

T_MIN = 1
T_MAX = 3
CAPACITIES = 5
M = 1_000
LEARNING_RATE = 0.01
