# PygameGraphRenderer.py
import pygame
import networkx as nx
import math
import sys
import os

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
        self.base_pos = {}  # Store original positions
        self.running = True
        
        # Zoom and pan variables
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Traffic flow animation toggle
        self.show_traffic_flow = show_traffic_flow

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
        """Draw updated node and edge states."""
        self.screen.fill((20, 20, 30))  # Dark blue-gray background for night road feel

        # Draw roads (edges) with traffic visualization
        import pygame.gfxdraw
        for (src, dst) in self.graph.edges():
            weight = edge_weights.get((src, dst), 0) + node_values.get((src, dst), 0)
            start = self.pos[src]
            end = self.pos[dst]
            
            # Road width based on traffic (simple linear scaling)
            min_width = 2
            max_width = int(12 * self.zoom)
            if weight == 0:
                road_width = min_width
            else:
                # Linear scaling from min to max based on weight
                normalized_weight = min(weight / 20.0, 1.0)  # Normalize to 0-1 range
                road_width = int(min_width + normalized_weight * (max_width - min_width))

            # Traffic intensity color (green = low traffic, yellow = medium, red = high)
            if weight == 0:
                traffic_color = (60, 60, 60)  # Dark gray for no traffic
            elif weight <= 5:
                # Green to yellow gradient (light traffic)
                intensity = weight / 5.0
                traffic_color = (int(50 + intensity * 155), int(150 + intensity * 105), 50)
            elif weight <= 15:
                # Yellow to red gradient (medium traffic)
                intensity = (weight - 5) / 10.0
                traffic_color = (int(205 + intensity * 50), int(255 - intensity * 100), 50)
            else:
                # High traffic - bright red
                traffic_color = (255, 50, 50)
            
            # Draw curved road with rounded ends (pill shape)
            self._draw_curved_road(start, end, road_width, traffic_color)
            
            # Add traffic flow animation (moving dots for high traffic) - toggleable
            if self.show_traffic_flow and weight > 3:
                self._draw_traffic_flow(start, end, weight, traffic_color)

        # Draw simple junction markers
        for z in self.zones:
            x, y = self.pos[z]
            
            # Simple junction marker - small white circle
            junction_radius = max(3, int(5 * self.zoom))
            pygame.gfxdraw.filled_circle(self.screen, x, y, junction_radius, (255, 255, 255))
            pygame.gfxdraw.aacircle(self.screen, x, y, junction_radius, (200, 200, 200))
            
            # Junction label
            font_size = max(8, int(12 * self.zoom))
            font = pygame.font.SysFont('Arial', font_size, bold=True)
            label_text = f"{z}"
            
            # Main label
            label = font.render(label_text, True, (255, 255, 255))
            label_x = x - label.get_width() // 2
            label_y = y - junction_radius - label.get_height() - 2
            
            # Add text shadow for better readability
            shadow = font.render(label_text, True, (0, 0, 0))
            self.screen.blit(shadow, (label_x + 1, label_y + 1))
            self.screen.blit(label, (label_x, label_y))

        # Add legend
        self._draw_legend()

        pygame.display.flip()
        self.clock.tick(30)
    
    def _draw_curved_road(self, start, end, width, color):
        """Draw a road with rounded ends (pill shape)."""
        import pygame.gfxdraw
        import math
        
        if width < 1:
            return
            
        x1, y1 = start
        x2, y2 = end
        
        # Ensure color is a proper tuple of integers
        color = tuple(int(c) for c in color[:3])  # Take only RGB, ensure integers
        
        # Calculate the distance and angle
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            # Single point - draw a circle
            radius = width // 2
            if radius > 0:
                pygame.gfxdraw.filled_circle(self.screen, x1, y1, radius, color)
                pygame.gfxdraw.aacircle(self.screen, x1, y1, radius, color)
            return
        
        # Normalize the direction vector
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Calculate perpendicular vector for width
        perp_x = -dy_norm * (width / 2)
        perp_y = dx_norm * (width / 2)
        
        # Calculate the four corners of the rectangle
        corners = [
            (int(x1 + perp_x), int(y1 + perp_y)),
            (int(x1 - perp_x), int(y1 - perp_y)),
            (int(x2 - perp_x), int(y2 - perp_y)),
            (int(x2 + perp_x), int(y2 + perp_y))
        ]
        
        # Draw the main rectangle body using regular pygame.draw for better compatibility
        pygame.draw.polygon(self.screen, color, corners)
        
        # Draw rounded ends (circles at both ends)
        radius = width // 2
        if radius > 0:
            # Start circle
            pygame.gfxdraw.filled_circle(self.screen, x1, y1, radius, color)
            pygame.gfxdraw.aacircle(self.screen, x1, y1, radius, color)
            
            # End circle
            pygame.gfxdraw.filled_circle(self.screen, x2, y2, radius, color)
            pygame.gfxdraw.aacircle(self.screen, x2, y2, radius, color)
    
    def _draw_traffic_flow(self, start, end, weight, color):
        """Draw animated traffic flow dots on roads."""
        import time
        import math
        
        # Calculate road direction and length
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0:
            return
            
        # Normalize direction
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Number of traffic dots based on weight
        num_dots = min(int(weight), 8)
        
        # Animate dots moving along the road
        time_offset = time.time() * 2  # Speed of animation
        
        for i in range(num_dots):
            # Position along the road (0 to 1)
            progress = ((time_offset + i * 0.3) % 2.0) / 2.0
            
            # Calculate dot position
            dot_x = int(start[0] + progress * dx)
            dot_y = int(start[1] + progress * dy)
            
            # Draw traffic dot
            dot_size = max(2, int(3 * self.zoom))
            pygame.gfxdraw.filled_circle(self.screen, dot_x, dot_y, dot_size, 
                                       (min(255, color[0] + 100), 
                                        min(255, color[1] + 100), 
                                        min(255, color[2] + 100)))
    
    def _draw_legend(self):
        """Draw a legend explaining the traffic visualization."""
        legend_x, legend_y = 10, 10
        font = pygame.font.SysFont('Arial', 12, bold=True)
        
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
