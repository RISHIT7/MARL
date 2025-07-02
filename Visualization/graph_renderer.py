import networkx as nx
import plotly.graph_objects as go
import math

class MaritimeTrafficGraph:
    def __init__(self, zones, valid_transitions, node_vmax=20, edge_vmax=20):
        self.G = nx.DiGraph()
        self.G.add_nodes_from(zones)
        self.G.add_edges_from(valid_transitions)

        self.node_vmax = node_vmax
        self.edge_vmax = edge_vmax

        self.node_values = {z: 0 for z in zones}
        self.edge_weights = {(u, v): 0 for u, v in valid_transitions}

        self.pos = nx.spring_layout(self.G, seed=42)
        self.frame = 0
        self.node_size_scale = 80 / math.sqrt(len(zones) + 1)

    def update(self, node_values, edge_weights):
        self.frame += 1
        self.node_values = node_values
        self.edge_weights = edge_weights

    def _node_color(self, value):
        normalized = min(value / self.node_vmax, 1.0)
        return f'rgba(100, 149, {int(255 * (0.3 + 0.7 * normalized))}, 1.0)'

    def _edge_color(self, weight):
        normalized = min(weight / self.edge_vmax, 1.0)
        if normalized < 0.5:
            r = int(255 * 2 * normalized)
            g = 255
        else:
            r = 255
            g = int(255 * (2 - 2 * normalized))
        return f'rgb({r},{g},0)'

    def create_figure(self,t):
        fig = go.Figure()

        # Edge lines
        for u, v in self.G.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            weight = self.edge_weights.get((u, v), 0)
            color = self._edge_color(weight)

            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                mode='lines',
                line=dict(width=4, color=color),
                hoverinfo='skip',
                showlegend=False
            ))

            # Midpoint label
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            fig.add_annotation(
                x=mid_x, y=mid_y,
                text=str(weight),
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,1)'
            )

		# Add invisible markers at edge midpoints for hover tooltips
        edge_hover_x, edge_hover_y, edge_hover_text = [], [], []
        for u, v in self.G.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            weight = self.edge_weights.get((u,v),0)

            edge_hover_x.append(mid_x)
            edge_hover_y.append(mid_y)
            edge_hover_text.append(f"Edge: {u} → {v}<br>Traffic Flow: {weight} cars")

        fig.add_trace(go.Scatter(
            x=edge_hover_x,
            y=edge_hover_y,
            mode='markers',
            marker=dict(size=10, color='rgba(0,0,0,0)'),  # Invisible
            hoverinfo='text',
            hovertext=edge_hover_text,
            showlegend=False
        ))
        # Node markers
        node_x, node_y, hover_text, sizes, colors = [], [], [], [], []
        for z in self.G.nodes():
            x, y = self.pos[z]
            value = self.node_values.get(z, 0)
            node_x.append(x)
            node_y.append(y)
            hover_text.append(f"{z}: {value} vessels")
            sizes.append(self.node_size_scale * (2 + value / self.node_vmax))
            colors.append(self._node_color(value))

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hovertext=hover_text,
            text=[f"{z}\n{self.node_values.get(z, 0)}" for z in self.G.nodes()],
            textposition="middle center",
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=10, color='white')
            ),
            showlegend=False
        ))

        fig.update_layout(
            title=f"Ship Traffic at Time {t}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            hovermode='closest',
        )
        return fig
    def reset(self):
        """Reset the graph to frame 0 and optionally reinitialize traffic values."""
        self.frame = 0
        for node in self.G.nodes():
            self.G.nodes[node]['value'] = 0
        for u, v in self.G.edges():
            self.G[u][v]['weight'] = 0