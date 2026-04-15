import pygame
import numpy as np
import math
import cv2
import colorsys

LOGICAL_SIZE = pygame.math.Vector2(20, 11.25)
SCREEN_SIZE = pygame.math.Vector2(1280, 720)
SCALE = SCREEN_SIZE.x / LOGICAL_SIZE.x

screen = None

num_particles = 200 
positions = []
predicted_positions = []
velocities = []

gravity = 10
collision_damping = 0.95
particle_color = (255, 255, 255)
particle_size = 0.05
particle_spacing = 0.05 

bounds_size = pygame.math.Vector2(8,5)
bounds_color = (100, 100, 100)

# SPH constants
smoothing_radius = 0.3 
particle_mass = 1 
densities = []

target_density = 1.5
pressure_multiplier = 30

max_display_speed = 2.0

# Interaction
interaction_radius = 2
interaction_strength = 50.0

def random_arrangment(half_bounds):
    for i in range(num_particles):
        x = np.random.uniform(-half_bounds.x, half_bounds.x)
        y = np.random.uniform(-half_bounds.y, half_bounds.y)
        positions.append(pygame.math.Vector2(x, y))
        velocities.append(pygame.math.Vector2(0, 0))

def grid_arrangment():
    particles_per_row = int(np.sqrt(num_particles))
    particles_per_col = (num_particles - 1) // particles_per_row + 1
    spacing = particle_size * 2 + particle_spacing
    
    for i in range(num_particles):
        x = (i % particles_per_row - particles_per_row / 2 + 0.5) * spacing
        y = (i // particles_per_row - particles_per_col / 2 + 0.5) * spacing
        positions.append(pygame.math.Vector2(x, y))
        velocities.append(pygame.math.Vector2(0, 0))

def start():
    global positions, velocities, densities, predicted_positions
    positions = []
    velocities = []
    densities = [0.0] * num_particles 
    
    predicted_positions = [pygame.math.Vector2(0, 0) for _ in range(num_particles)]

    half_bounds = bounds_size / 2 - pygame.math.Vector2(particle_size, particle_size)

    # random_arrangment(half_bounds)
    grid_arrangment()

""" 
SPH Functions
"""
def smoothing_kernel(radius, distance):
    if distance >= radius:
        return 0.0
    
    volume = (math.pi * (radius ** 4.0)) / 6.0
    return ((radius - distance) ** 2.0) / volume

def smoothing_kernel_derivative(radius, distance):
    if distance >= radius:
        return 0.0
    
    scale = 12.0 / ((radius ** 4.0) * math.pi)
    return (distance - radius) * scale

def convert_density_to_pressure(density):
    density_error = density - target_density
    pressure = density_error * pressure_multiplier
    return max(0.0, pressure)  # Clamp to zero to prevent vacuum attraction

def calculate_shared_pressure(density_a, density_b):
    pressure_a = convert_density_to_pressure(density_a)
    pressure_b = convert_density_to_pressure(density_b)
    return (pressure_a + pressure_b) / 2.0

def calculate_densities():
    for i in range(num_particles):
        density = 0.0
        for j in range(num_particles):
            # Use predicted positions for distance
            dist = predicted_positions[i].distance_to(predicted_positions[j])
            influence = smoothing_kernel(smoothing_radius, dist)
            density += particle_mass * influence
        densities[i] = density

def calculate_pressure_force(particle_index):
    pressure_force = pygame.math.Vector2(0, 0)
    density_self = densities[particle_index] 
    
    for other_particle_index in range(num_particles):
        if particle_index == other_particle_index:
            continue
            
        # Use predicted positions for direction and distance
        offset = predicted_positions[other_particle_index] - predicted_positions[particle_index]
        dst = offset.magnitude()
        
        if dst == 0:
            dir = pygame.math.Vector2(np.random.uniform(-1, 1), np.random.uniform(-1, 1))
            if dir.length() > 0:
                dir = dir.normalize()
            else:
                dir = pygame.math.Vector2(0, 1)
        else:
            dir = offset / dst
            
        slope = smoothing_kernel_derivative(smoothing_radius, dst)
        density_other = densities[other_particle_index]
        
        if density_other == 0:
            continue
            
        shared_pressure = calculate_shared_pressure(density_self, density_other)
        pressure_force += dir * shared_pressure * slope * particle_mass / density_other
        
    return pressure_force

"""
Rendering
"""

def velocity_to_color(speed, max_speed):
    t = speed / max_speed if max_speed > 0 else 0.0
    t = max(0.0, min(1.0, t))
    hue = (1.0 - t) * 240.0 / 360.0  # blue (240°) -> green -> yellow -> red (0°)
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


kernel_sprite = None

def create_kernel_cloud():
    radius_screen = int(smoothing_radius * SCALE)
    size = radius_screen * 2
    
    surf = pygame.Surface((size, size))
    center = pygame.math.Vector2(radius_screen, radius_screen)
    
    max_influence = smoothing_kernel(smoothing_radius, 0)
    
    for x in range(size):
        for y in range(size):
            dist_screen = center.distance_to(pygame.math.Vector2(x, y))
            dist_logical = dist_screen / SCALE
            
            if dist_logical < smoothing_radius:
                influence = smoothing_kernel(smoothing_radius, dist_logical)
                
                intensity = (influence / max_influence) if max_influence > 0 else 0
                
                r = 0
                g = int(intensity * 15)
                b = int(intensity * 50)
                
                surf.set_at((x, y), (r, g, b))
                
    return surf

def calculate_interaction_force(input_pos, radius, strength, particle_index):
    offset = input_pos - positions[particle_index]
    sqr_dst = offset.dot(offset)
    if sqr_dst >= radius * radius:
        return pygame.math.Vector2(0, 0)
    dst = math.sqrt(sqr_dst)
    dir_to_input = offset / dst if dst > 1e-6 else pygame.math.Vector2(0, 0)
    centre_t = 1.0 - dst / radius
    return (dir_to_input * strength - velocities[particle_index]) * centre_t

def resolve_collisions(pos, vel):
    half_bounds = bounds_size / 2 - pygame.math.Vector2(particle_size, particle_size)

    if abs(pos.x) > half_bounds.x:
        pos.x = half_bounds.x * np.sign(pos.x)
        vel.x *= -1 * collision_damping

    if abs(pos.y) > half_bounds.y:
        pos.y = half_bounds.y * np.sign(pos.y)
        vel.y *= -1 * collision_damping

def update(dt, mouse_input_pos=None, mouse_strength=0.0):
    global screen, positions, predicted_positions, velocities, gravity

    # Predict next positions BEFORE calculating densities
    prediction_time_step = 1.0 / 120.0
    for i in range(num_particles):
        velocities[i] += pygame.math.Vector2(0, 1) * gravity * dt
        predicted_positions[i] = positions[i] + velocities[i] * prediction_time_step

    calculate_densities()

    new_velocities = list(velocities)
    for i in range(num_particles):
        pressure_force = calculate_pressure_force(i)

        if densities[i] > 0:
            pressure_acceleration = pressure_force / densities[i]
            new_velocities[i] += pressure_acceleration * dt

        if mouse_input_pos is not None and mouse_strength != 0.0:
            interaction = calculate_interaction_force(mouse_input_pos, interaction_radius, mouse_strength, i)
            new_velocities[i] += interaction * dt

    velocities = new_velocities

    for i in range(num_particles):
        positions[i] += velocities[i] * dt
        resolve_collisions(positions[i], velocities[i])

        draw_pos = (positions[i] + LOGICAL_SIZE / 2) * SCALE

        # cloud_rect = kernel_sprite.get_rect(center=(int(draw_pos.x), int(draw_pos.y)))
        # screen.blit(kernel_sprite, cloud_rect, special_flags=pygame.BLEND_RGB_ADD)

        color = velocity_to_color(velocities[i].magnitude(), max_display_speed)
        pygame.draw.circle(screen, color, draw_pos, particle_size * SCALE)

VIDEO_FPS = 30
VIDEO_OUTPUT = "simulation.mp4"

def main():
    global screen, kernel_sprite
    pygame.init()
    start()
    screen = pygame.display.set_mode((int(SCREEN_SIZE.x), int(SCREEN_SIZE.y)))

    # kernel_sprite = create_kernel_cloud()

    clock = pygame.time.Clock()
    background_color = (0, 0, 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, VIDEO_FPS, (int(SCREEN_SIZE.x), int(SCREEN_SIZE.y)))
    video_frame_accum = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(background_color)

        center = SCREEN_SIZE / 2
        bounds_screen = bounds_size * SCALE
        bounds_rect = pygame.Rect(
            int(center.x - bounds_screen.x / 2),
            int(center.y - bounds_screen.y / 2),
            int(bounds_screen.x),
            int(bounds_screen.y)
        )
        pygame.draw.rect(screen, bounds_color, bounds_rect, 2)

        # Mouse interaction
        mouse_screen = pygame.math.Vector2(pygame.mouse.get_pos())
        mouse_logical = mouse_screen / SCALE - LOGICAL_SIZE / 2
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0]:       # left click: attract
            mouse_strength = interaction_strength
        elif mouse_buttons[2]:     # right click: repel
            mouse_strength = -interaction_strength
        else:
            mouse_strength = 0.0

        # Draw interaction circle when mouse button is held
        if mouse_strength != 0.0:
            circle_color = (80, 180, 255) if mouse_strength > 0 else (255, 100, 80)
            pygame.draw.circle(screen, circle_color, (int(mouse_screen.x), int(mouse_screen.y)),
                               int(interaction_radius * SCALE), 2)

        clock.tick(VIDEO_FPS)
        fixed_dt = 0.005
        for _ in range(4):
            update(fixed_dt, mouse_logical if mouse_strength != 0.0 else None, mouse_strength)

        font = pygame.font.SysFont("Arial", 10)
        text = font.render(f"FPS: {clock.get_fps():.2f}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        pygame.display.flip()

        video_frame_accum += 1.0 / VIDEO_FPS
        if video_frame_accum >= 1.0 / VIDEO_FPS:
            video_frame_accum -= 1.0 / VIDEO_FPS
            frame = pygame.surfarray.array3d(screen)        # (W, H, 3) RGB
            frame = np.transpose(frame, (1, 0, 2))          # -> (H, W, 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)

    video.release()
    pygame.quit()
    print(f"Video saved to {VIDEO_OUTPUT}")

if __name__ == "__main__":
    main()