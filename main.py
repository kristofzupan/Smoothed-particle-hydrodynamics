import pygame
import numpy as np
import math

LOGICAL_SIZE = pygame.math.Vector2(20, 11.25)
SCREEN_SIZE = pygame.math.Vector2(1280, 720)
SCALE = SCREEN_SIZE.x / LOGICAL_SIZE.x

screen = None

num_particles = 288
positions = []
velocities = []

gravity = 0
collision_damping = 0.7
particle_color = (255, 255, 255)
particle_size = 0.05
particle_spacing = 0.2 

bounds_size = pygame.math.Vector2(19, 10.25)
bounds_color = (100, 100, 100)

# SPH constants
smoothing_radius = 1 # 'h' in the formulas
particle_mass = 1 # 'm' in the formulas
densities = []

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
    global positions, velocities, densities
    positions = []
    velocities = []
    densities = [0.0] * num_particles # Initialize with zeros

    half_bounds = bounds_size / 2 - pygame.math.Vector2(particle_size, particle_size)

    random_arrangment(half_bounds)
    # grid_arrangment(half_bounds)

def smoothing_kernel(radius, distance):
    if distance >= radius:
        return 0.0
    
    volume = (math.pi * radius ** 8.0) / 4.0
    return ((radius * radius - distance * distance) ** 3.0) / volume

def calculate_densities():
    for i in range(num_particles):
        density = 0.0
        for j in range(num_particles):
            # Calculate distance between particle i and j
            dist = positions[i].distance_to(positions[j])
            
            # Get influence from kernel
            influence = smoothing_kernel(smoothing_radius, dist)
            density += particle_mass * influence
            
        densities[i] = density

kernel_sprite = None

def create_kernel_cloud():
    # Convert logical radius to screen pixels
    radius_screen = int(smoothing_radius * SCALE)
    size = radius_screen * 2
    
    # Create a black surface (Black = 0, so it doesn't affect additive blending)
    surf = pygame.Surface((size, size))
    center = pygame.math.Vector2(radius_screen, radius_screen)
    
    # Find the maximum possible value of the kernel (at distance 0)
    max_influence = smoothing_kernel(smoothing_radius, 0)
    
    for x in range(size):
        for y in range(size):
            dist_screen = center.distance_to(pygame.math.Vector2(x, y))
            dist_logical = dist_screen / SCALE
            
            if dist_logical < smoothing_radius:
                influence = smoothing_kernel(smoothing_radius, dist_logical)
                
                # Normalize the influence and map to a dark blue color
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
    global screen, positions, velocities, gravity

    calculate_densities()

    for i in range(len(positions)):
        # (Later: Pressure force calculation will go here)

        velocities[i] += pygame.math.Vector2(0, 1) * gravity * dt
        positions[i] += velocities[i] * dt
        resolve_collisions(positions[i], velocities[i])

        # Convert logical position to screen coordinates
        draw_pos = (positions[i] + LOGICAL_SIZE / 2) * SCALE
        
        # --- REPLACE THE DRAW.CIRCLE WITH THIS ---
        # 1. Get the rect so we center the cloud perfectly on the particle
        cloud_rect = kernel_sprite.get_rect(center=(int(draw_pos.x), int(draw_pos.y)))
        # 2. Blit the cloud using additive blending
        screen.blit(kernel_sprite, cloud_rect, special_flags=pygame.BLEND_RGB_ADD)
        # 3. Draw a tiny white dot in the center to show the actual particle        
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

        # Draw bounding box (convert logical bounds to screen coordinates)
        center = SCREEN_SIZE / 2
        bounds_screen = bounds_size * SCALE
        bounds_rect = pygame.Rect(
            int(center.x - bounds_screen.x / 2),
            int(center.y - bounds_screen.y / 2),
            int(bounds_screen.x),
            int(bounds_screen.y)
        )
        pygame.draw.rect(screen, bounds_color, bounds_rect, 2)

        dt = min(clock.tick(0) / 1000.0, 0.05)  # Delta time in seconds, capped at 50ms
        update(dt)

        font = pygame.font.SysFont("Arial", 10)
        text = font.render( f"FPS: {clock.get_fps():.2f}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        pygame.display.flip()


    pygame.quit()

if __name__ == "__main__":
    main()