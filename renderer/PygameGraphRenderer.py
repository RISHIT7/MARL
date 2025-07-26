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
        """Automatically assign node positions using NetworkX's spring layout."""
        self.base_pos = {}
        self.pos = {}

        # Compute layout using spring_layout
        layout = nx.spring_layout(
            self.graph,
            scale=min(self.width, self.height) * 0.4,  # fit within window size
            center=(self.width // 2, self.height // 2),
            seed=42  # for reproducibility
        )

        # Store computed positions
        for z in self.zones:
            x, y = layout[z]
            self.base_pos[z] = (int(x), int(y))
            self.pos[z] = (int(x), int(y))

    
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
        self.screen.fill((0, 0, 0))  # black background

        # Draw edges
        import pygame.gfxdraw
        for (src, dst) in self.graph.edges():
            weight = edge_weights.get((src, dst), 0)
            start = self.pos[src]
            end = self.pos[dst]
            line_width = max(1, int(1 + weight // 5))
            # Draw anti-aliased lines for edges (fallback to normal line for width>1)
            # if line_width == 1:
            pygame.gfxdraw.line(self.screen, start[0], start[1], end[0], end[1], (200, 200, 200))
            # else:
            #     pygame.draw.line(self.screen, (200, 200, 200), start, end, line_width)

        # Draw nodes
        for z in self.zones:
            x, y = self.pos[z]
            count = node_values.get(z, 0)
            color = (100, 100, 255)
            # Scale font with zoom
            font_size = max(12, int(20 * self.zoom))
            font = pygame.font.SysFont(None, font_size)
            label_text = f"{z}:{count}"
            label = font.render(label_text, True, (255, 255, 255))  # white text
            # Calculate radius based on text size with padding
            padding = 10
            text_width, text_height = label.get_width(), label.get_height()
            radius = int(max(text_width, text_height) / 2 + padding)
            # Draw node as a circle with only a colored boundary (outline), inside is black, using anti-aliased gfxdraw
            pygame.gfxdraw.filled_circle(self.screen, x, y, radius, (0, 0, 0))  # fill with black
            pygame.gfxdraw.aacircle(self.screen, x, y, radius, (255,255,255))  # anti-aliased outline
            pygame.gfxdraw.aacircle(self.screen, x, y, radius-1, (255,255,255))  # thicker outline
            # Draw label centered
            label_x = x - text_width // 2
            label_y = y - text_height // 2
            self.screen.blit(label, (label_x, label_y))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()
