from environment.TrafficEnvMain import MaritimeTrafficEnv
from policy.RandomPolicy import RandomPolicy
import pygame
from renderer.PygameGraphRenderer import PygameGraphRenderer

import time

if __name__ == "__main__":
    env = MaritimeTrafficEnv()
    agent = RandomPolicy(env)
    renderer = PygameGraphRenderer(
        zones=env.zones, 
        valid_transitions=env.valid_transitions, 
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
            print(f"Step completed - Reward: {reward}, Terminated: {terminated}")

        # Update renderer every frame for smooth visuals
        try:
            # Check for common environment state attributes
            if hasattr(env, 'state') and env.state is not None:
                # Use environment state if available
                traffic_data = getattr(env.state, 'flows', {}) if hasattr(env.state, 'flows') else {}
                edge_data = getattr(env.state, 'capacities', {}) if hasattr(env.state, 'capacities') else {}
                renderer.update(traffic_data, edge_data)
            elif hasattr(env, 'current_traffic'):
                renderer.update(env.current_traffic, getattr(env, 'edge_weights', {}))
            elif hasattr(env, 'n_tot'):
                renderer.update(env.n_tot, getattr(env, 'edge_weight', {}))
            else:
                # Fallback with empty data
                renderer.update({}, {})
        except Exception as e:
            print(f"Renderer update error: {e}")
            # Continue without crashing
            pass

    renderer.close()
    if hasattr(env, 'close'):
        env.close()
