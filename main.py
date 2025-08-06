from environment.TrafficEnvMain import RoadTrafficEnv
from policy.RandomPolicy import RandomPolicy
import pygame
from renderer.PygameGraphRenderer import PygameGraphRenderer

import time

if __name__ == "__main__":
    env = RoadTrafficEnv()
    agent = RandomPolicy(env)
    
    # Create junction-to-junction transitions for renderer compatibility
    junction_transitions = []
    for i in env.arrival_junctions:
        junction_transitions.append(("j_source", i))
    
    # Convert road connections back to junction connections for visualization
    for road in env.roads:
        if not road.startswith("road_source"):
            if "_to_" in road:
                parts = road.replace("road_", "").split("_to_")
                if len(parts) == 2:
                    src, dest = parts
                    junction_transitions.append((f"j_{src}", f"j_{dest}"))
    
    renderer = PygameGraphRenderer(
        zones=env.junctions,  # Use junctions instead of zones
        valid_transitions=junction_transitions,  # Junction-to-junction transitions
        width=1000, 
        height=1000, 
        show_traffic_flow=False)
    
    print("Interactive Controls:")
    print("  Mouse wheel - Zoom in/out")
    print("  Left click + drag - Pan around")
    print("  Close window to quit")
    print("  Simulation updates automatically every 1 second")

    obs, _ = env.reset(seed=1)
    terminated = False
    truncated = False
    
    # Create a clock for controlling frame rate
    clock = pygame.time.Clock()
    
    # Zoom and pan variables
    zoom_level = 1.0
    pan_x, pan_y = 0, 0
    dragging = False
    last_mouse_pos = (0, 0)
    
    # Auto-update timing
    last_step_time = time.time()
    step_interval = 1.0  # Update every 1 second

    while not terminated and not truncated:
        current_time = time.time()
        dt = clock.tick(60) / 1000.0  # 60 FPS for smooth interaction

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    dragging = True
                    last_mouse_pos = event.pos
                elif event.button == 4:  # Mouse wheel up (zoom in)
                    zoom_level = min(zoom_level * 1.1, 5.0)
                    if hasattr(renderer, 'set_zoom'):
                        renderer.set_zoom(zoom_level)
                elif event.button == 5:  # Mouse wheel down (zoom out)
                    zoom_level = max(zoom_level / 1.1, 0.2)
                    if hasattr(renderer, 'set_zoom'):
                        renderer.set_zoom(zoom_level)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    # Calculate pan offset
                    mouse_x, mouse_y = event.pos
                    dx = mouse_x - last_mouse_pos[0]
                    dy = mouse_y - last_mouse_pos[1]
                    pan_x += dx
                    pan_y += dy
                    if hasattr(renderer, 'set_pan'):
                        renderer.set_pan(pan_x, pan_y)
                    last_mouse_pos = event.pos

        # Auto-update simulation at specified interval
        if (current_time - last_step_time) >= step_interval:
            action = agent.select_action()
            obs, reward, terminated, truncated, info = env.step(action)
            last_step_time = current_time
            print(f"Step {env.t} - Reward: {reward:.2f}, Terminated: {terminated}")
            
            # Print road traffic information
            total_vehicles = sum(env.n_road_tot.values())
            print(f"Total vehicles on roads: {total_vehicles}")
            if total_vehicles > 0:
                print("Road traffic:")
                for road, count in env.n_road_tot.items():
                    if count > 0:
                        capacity = env.road_capacities[road]
                        utilization = (count / capacity) * 100 if capacity > 0 else 0
                        print(f"  {road}: {count}/{capacity} vehicles ({utilization:.1f}%)")

        # Update renderer every frame for smooth visuals
        try:
            # Convert road-based data to junction-based data for renderer
            junction_data = {}
            edge_data = {}
            
            # Since we removed junction tracking, use empty junction data
            # Set all junctions to 0 vehicles
            for junction in env.junctions:
                junction_data[junction] = 0
            
            # Convert road traffic to edge weights for visualization
            for road, count in env.n_road_tot.items():
                if road.startswith("road_source_to_"):
                    dest = road.split('_')[-1]
                    edge_data[("j_source", f"j_{dest}")] = count
                elif "_to_" in road:
                    parts = road.replace("road_", "").split("_to_")
                    if len(parts) == 2:
                        src, dest = parts
                        edge_data[(f"j_{src}", f"j_{dest}")] = count
            
            renderer.update(junction_data, edge_data)
            
        except Exception as e:
            print(f"Renderer update error: {e}")
            # Continue without crashing
            pass

    renderer.close()
    if hasattr(env, 'close'):
        env.close()
