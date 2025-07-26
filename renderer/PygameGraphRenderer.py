# PygameGraphRenderer.py
import pygame
import networkx as nx
import math
import sys
import os
import time

# Add the parent directory to the path so we can import from environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.parameters import M

class PygameGraphRenderer:
    def __init__(self, zones, valid_transitions, width=800, height=800, show_traffic_flow=True):
        pygame.init()
        self.width, self.height = width, height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Maritime Traffic Viewer")
        self.clock = pygame.time.Clock()
        self.graph = nx.DiGraph()
        self.zones = zones
        self.pos = {}
        self.base_pos = {}
        self.running = True
        # Zoom and pan variables
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        # Traffic flow animation toggle
        self.show_traffic_flow = show_traffic_flow
        # Font cache for performance
        self.font_cache = {}
        # Build graph structure
        self.graph.add_nodes_from(zones)
        self.graph.add_edges_from(valid_transitions)
        self._layout_nodes()

    def _layout_nodes(self):
        """Assign node positions. Draw dummy node above, others in grid."""
        self.base_pos = {}
        self.pos = {}
        num_nodes = len(self.zones)
        if num_nodes >= 100:
            # Draw dummy node (assume first in self.zones) above the grid
            margin = 40
            grid_nodes = self.zones[1:]
            grid_cols = math.ceil(math.sqrt(len(grid_nodes)))
            grid_rows = math.ceil(len(grid_nodes) / grid_cols)
            grid_width = self.width - 2 * margin
            grid_height = self.height - 2 * margin
            cell_w = grid_width // max(1, grid_cols-1) if grid_cols > 1 else grid_width
            cell_h = grid_height // max(1, grid_rows-1) if grid_rows > 1 else grid_height
            # Dummy node centered above grid
            dummy_x = self.width // 2
            dummy_y = margin // 2
            self.base_pos[self.zones[0]] = (dummy_x, dummy_y)
            self.pos[self.zones[0]] = (dummy_x, dummy_y)
            # Grid nodes
            for idx, z in enumerate(grid_nodes):
                row = idx // grid_cols
                col = idx % grid_cols
                x = margin + col * cell_w
                y = margin + row * cell_h + margin  # leave space for dummy node
                self.base_pos[z] = (int(x), int(y))
                self.pos[z] = (int(x), int(y))
        else:
            # Use spring layout for small/medium graphs
            layout = nx.spring_layout(
                self.graph,
                scale=min(self.width, self.height) * 0.4,
                center=(self.width // 2, self.height // 2),
                seed=42
            )
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
    
    def toggle_traffic_flow(self):
        """Toggle the traffic flow animation on/off."""
        self.show_traffic_flow = not self.show_traffic_flow
        return self.show_traffic_flow
    
    def set_traffic_flow(self, enabled):
        """Set the traffic flow animation on/off."""
        self.show_traffic_flow = enabled
    
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
        """Draw updated node and edge states. Further optimized for large graphs."""
        screen = self.screen
        zones = self.zones
        pos = self.pos
        zoom = self.zoom
        show_traffic_flow = self.show_traffic_flow
        node_count = len(zones)
        screen.fill((20, 20, 30))
        # Precompute edge drawing mode
        use_straight_lines = node_count > 200
        # Precompute font for nodes
        font_size = max(8, int(12 * zoom))
        font = self._get_font(font_size)
        # Draw edges
        min_width = 2
        max_width = int(12 * zoom)
        for (src, dst) in self.graph.edges():
            weight = edge_weights.get((src, dst), 0) + node_values.get((src, dst), 0)
            start = pos[src]
            end = pos[dst]
            if weight == 0:
                road_width = min_width
            else:
                normalized_weight = min(weight / 20.0, 1.0)
                road_width = int(min_width + normalized_weight * (max_width - min_width))
            # Traffic intensity color
            if weight == 0:
                traffic_color = (60, 60, 60)
            elif weight <= 5:
                intensity = weight / 5.0
                traffic_color = (int(50 + intensity * 155), int(150 + intensity * 105), 50)
            elif weight <= 15:
                intensity = (weight - 5) / 10.0
                traffic_color = (int(205 + intensity * 50), int(255 - intensity * 100), 50)
            else:
                traffic_color = (255, 50, 50)
            if use_straight_lines:
                pygame.draw.line(screen, traffic_color, start, end, max(1, road_width))
            else:
                self._draw_curved_road(start, end, road_width, traffic_color)
            if show_traffic_flow and weight > 3 and not use_straight_lines:
                self._draw_traffic_flow(start, end, weight, traffic_color)
        # Draw nodes and labels
        junction_radius = max(3, int(5 * zoom))
        for z in zones:
            x, y = pos[z]
            pygame.draw.circle(screen, (255, 255, 255), (x, y), junction_radius)
            pygame.draw.circle(screen, (200, 200, 200), (x, y), junction_radius, 1)
            label_text = str(z)
            label = font.render(label_text, True, (255, 255, 255))
            label_x = x - label.get_width() // 2
            label_y = y - junction_radius - label.get_height() - 2
            shadow = font.render(label_text, True, (0, 0, 0))
            screen.blit(shadow, (label_x + 1, label_y + 1))
            screen.blit(label, (label_x, label_y))
        self._draw_legend(font)
        pygame.display.flip()
        self.clock.tick(30)

    def _get_font(self, size):
        if size not in self.font_cache:
            self.font_cache[size] = pygame.font.SysFont('Arial', size, bold=True)
        return self.font_cache[size]
    
    def _draw_curved_road(self, start, end, width, color):
        """Draw a road with rounded ends (pill shape)."""
        if width < 1:
            return
        x1, y1 = start
        x2, y2 = end
        color = tuple(int(c) for c in color[:3])
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            radius = width // 2
            if radius > 0:
                pygame.draw.circle(self.screen, color, (x1, y1), radius)
            return
        dx_norm = dx / length
        dy_norm = dy / length
        perp_x = -dy_norm * (width / 2)
        perp_y = dx_norm * (width / 2)
        corners = [
            (int(x1 + perp_x), int(y1 + perp_y)),
            (int(x1 - perp_x), int(y1 - perp_y)),
            (int(x2 - perp_x), int(y2 - perp_y)),
            (int(x2 + perp_x), int(y2 + perp_y))
        ]
        pygame.draw.polygon(self.screen, color, corners)
        radius = width // 2
        if radius > 0:
            pygame.draw.circle(self.screen, color, (x1, y1), radius)
            pygame.draw.circle(self.screen, color, (x2, y2), radius)
    
    def _draw_traffic_flow(self, start, end, weight, color):
        """Draw animated traffic flow dots on roads."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length == 0:
            return
        dx_norm = dx / length
        dy_norm = dy / length
        num_dots = min(int(weight), 8)
        time_offset = time.time() * 2
        for i in range(num_dots):
            progress = ((time_offset + i * 0.3) % 2.0) / 2.0
            dot_x = int(start[0] + progress * dx)
            dot_y = int(start[1] + progress * dy)
            dot_size = max(2, int(3 * self.zoom))
            dot_color = (
                min(255, color[0] + 100),
                min(255, color[1] + 100),
                min(255, color[2] + 100)
            )
            pygame.draw.circle(self.screen, dot_color, (dot_x, dot_y), dot_size)
    
    def _draw_legend(self, font=None):
        """Draw a legend explaining the traffic visualization."""
        legend_x, legend_y = 10, 10
        if font is None:
            font = self._get_font(12)
        # Background for legend
        legend_bg = pygame.Surface((200, 100))
        legend_bg.fill((0, 0, 0))
        legend_bg.set_alpha(180)
        self.screen.blit(legend_bg, (legend_x, legend_y))
        # Legend text
        legend_items = [
            ("Traffic Legend:", (255, 255, 255)),
            ("Green roads: Light traffic", (100, 255, 100)),
            ("Yellow roads: Medium traffic", (255, 255, 100)),
            ("Red roads: Heavy traffic", (255, 100, 100)),
            ("Moving dots: Active flow", (150, 150, 255))
        ]
        for i, (text, color) in enumerate(legend_items):
            label = font.render(text, True, color)
            self.screen.blit(label, (legend_x + 5, legend_y + 5 + i * 18))

    def close(self):
        pygame.quit()
