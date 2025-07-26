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
        self.base_pos = {}  # Store original positions
        self.running = True
        
        # Zoom and pan variables
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

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
            self.base_pos[z] = (x, y)  # Store base position
            self.pos[z] = (x, y)  # Current transformed position
    
    def set_zoom(self, zoom_level):
        """Set the zoom level and update node positions."""
        self.zoom = zoom_level
        self._update_positions()
    
    def set_pan(self, pan_x, pan_y):
        """Set the pan offset and update node positions."""
        self.pan_x = pan_x
        self.pan_y = pan_y
        self._update_positions()
    
    def _update_positions(self):
        """Update all node positions based on current zoom and pan."""
        center_x, center_y = self.width // 2, self.height // 2
        for z in self.zones:
            base_x, base_y = self.base_pos[z]
            # Apply zoom around center
            zoomed_x = center_x + (base_x - center_x) * self.zoom
            zoomed_y = center_y + (base_y - center_y) * self.zoom
            # Apply pan
            final_x = zoomed_x + self.pan_x
            final_y = zoomed_y + self.pan_y
            self.pos[z] = (int(final_x), int(final_y))

    def update(self, node_values, edge_weights):
        """Draw updated node and edge states."""
        self.screen.fill((255, 255, 255))  # white background

        # Draw edges
        for (src, dst) in self.graph.edges():
            weight = edge_weights.get((src, dst), 0)
            start = self.pos[src]
            end = self.pos[dst]
            line_width = max(1, int(1 + weight // 5))
            pygame.draw.line(self.screen, (200, 200, 200), start, end, line_width)

        # Draw nodes
        for z in self.zones:
            x, y = self.pos[z]
            count = node_values.get(z, 0)
            color = (100, 100, 255)
            radius = max(10, int(20 * self.zoom))  # Scale node size with zoom
            pygame.draw.circle(self.screen, color, (x, y), radius)
            
            # Scale font with zoom
            font_size = max(12, int(20 * self.zoom))
            font = pygame.font.SysFont(None, font_size)
            label = font.render(f"{z}:{count}", True, (0, 0, 0))
            label_x = x - label.get_width() // 2
            label_y = y - label.get_height() // 2
            self.screen.blit(label, (label_x, label_y))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
