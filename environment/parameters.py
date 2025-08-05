HORIZON = 100
W_R = 1.0
W_D = 1.0

# Grid network parameters
GRID_WIDTH = 4
GRID_HEIGHT = 4
NODES = GRID_WIDTH * GRID_HEIGHT  # 16 nodes in a 4x4 grid

# Arrival nodes (top row)
ARRIVAL_NODES = [1, 2, 3, 4]

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
TERMINAL_NODES = [13, 14, 15, 16]

# Simple arrival distribution - equal probability for each arrival node and time step
ARRIVAL_DIST = {}
total_entries = len(ARRIVAL_NODES) * 4  # 4 nodes * 4 time steps = 16 entries
probability_per_entry = 1.0 / total_entries

for node in ARRIVAL_NODES:
    for time_step in range(1, 5):
        ARRIVAL_DIST[(node, time_step)] = probability_per_entry

T_MIN = 1
T_MAX = 3
CAPACITIES = 5
M = 1_000
LEARNING_RATE = 0.01
