# PygameGraphRenderer.py
import pygame
import networkx as nx
import math

class PygameGraphRenderer:
    def __init__(self, zones, valid_transitions, width=800, height=800):
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maritime Traffic Viewer")
        self.clock = pygame.time.Clock()
        self.graph = nx.DiGraph()
        self.zones = zones
        self.pos = {}
        self.running = True

        # Build graph structure
        self.graph.add_nodes_from(zones)
        self.graph.add_edges_from(valid_transitions)
        self._layout_nodes()

    def _layout_nodes(self):
        """Assign fixed positions for nodes using circular layout."""
        angle_step = 2 * math.pi / len(self.zones)
        center_x, center_y = self.width // 2, self.height // 2
        radius = 250
        for i, z in enumerate(self.zones):
            angle = i * angle_step
            x = center_x + int(radius * math.cos(angle))
            y = center_y + int(radius * math.sin(angle))
            self.pos[z] = (x, y)

    def update(self, node_values, edge_weights):
        """Draw updated node and edge states."""
        self.screen.fill((255, 255, 255))  # white background

        # Draw edges
        for (src, dst) in self.graph.edges():
            weight = edge_weights.get((src, dst), 0)
            start = self.pos[src]
            end = self.pos[dst]
            pygame.draw.line(self.screen, (200, 200, 200), start, end, 1 + weight // 5)

        # Draw nodes
        for z in self.zones:
            x, y = self.pos[z]
            count = node_values.get(z, 0)
            color = (100, 100, 255)
            pygame.draw.circle(self.screen, color, (x, y), 20)
            font = pygame.font.SysFont(None, 20)
            label = font.render(f"{z}:{count}", True, (0, 0, 0))
            self.screen.blit(label, (x - 20, y - 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
