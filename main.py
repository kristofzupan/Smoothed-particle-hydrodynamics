import pygame
import numpy as np
import math

LOGICAL_SIZE = pygame.math.Vector2(20, 11.25)
SCREEN_SIZE = pygame.math.Vector2(1280, 720)
SCALE = SCREEN_SIZE.x / LOGICAL_SIZE.x

screen = None

num_particles = 400 
positions = []
predicted_positions = []
velocities = []

gravity = 0 
collision_damping = 0.95
particle_color = (255, 255, 255)
particle_size = 0.05
particle_spacing = 0.1 

bounds_size = pygame.math.Vector2(6,4)
bounds_color = (100, 100, 100)

# --- SPH constants ---
smoothing_radius = 0.3 
particle_mass = 1 
densities = []

target_density = 1.5       
pressure_multiplier = 5

def random_arrangment(half_bounds):
    for i in range(num_particles):
        x = np.random.uniform(-half_bounds.x, half_bounds.x)
        y = np.random.uniform(-half_bounds.y, half_bounds.y)
        positions.append(pygame.math.Vector2(x, y))
        velocities.append(pygame.math.Vector2(0, 0))

def grid_arrangment(half_bounds):
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
    grid_arrangment(half_bounds)

# --- SPH Math Functions ---

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
    return pressure

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

# --- Rendering ---

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

def resolve_collisions(pos, vel):
    half_bounds = bounds_size / 2 - pygame.math.Vector2(particle_size, particle_size)

    if abs(pos.x) > half_bounds.x:
        pos.x = half_bounds.x * np.sign(pos.x)
        vel.x *= -1 * collision_damping

    if abs(pos.y) > half_bounds.y:
        pos.y = half_bounds.y * np.sign(pos.y)
        vel.y *= -1 * collision_damping

def update(dt):
    global screen, positions, predicted_positions, velocities, gravity

    # 5. Predict next positions BEFORE calculating densities
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

    velocities = new_velocities

    for i in range(num_particles):
        positions[i] += velocities[i] * dt
        resolve_collisions(positions[i], velocities[i])

        draw_pos = (positions[i] + LOGICAL_SIZE / 2) * SCALE
        
        cloud_rect = kernel_sprite.get_rect(center=(int(draw_pos.x), int(draw_pos.y)))
        screen.blit(kernel_sprite, cloud_rect, special_flags=pygame.BLEND_RGB_ADD)
        
        pygame.draw.circle(screen, particle_color, draw_pos, particle_size * SCALE)

def main():
    global screen, kernel_sprite
    pygame.init()
    start()
    screen = pygame.display.set_mode((int(SCREEN_SIZE.x), int(SCREEN_SIZE.y)))

    kernel_sprite = create_kernel_cloud()

    clock = pygame.time.Clock()

    background_color = (0, 0, 0)

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

        dt = min(clock.tick(0) / 1000.0, 0.05)  
        update(dt)

        font = pygame.font.SysFont("Arial", 10)
        text = font.render( f"FPS: {clock.get_fps():.2f}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()