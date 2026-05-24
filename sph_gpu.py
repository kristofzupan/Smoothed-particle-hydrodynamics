import taichi as ti
import numpy as np
import math
import time

ti.init(arch=ti.metal)

LOGICAL_SIZE_X = 20.0
LOGICAL_SIZE_Y = 11.25
SCREEN_W = 1920
SCREEN_H = 1080

num_particles = 1000
gravity = 10
collision_damping = 0.9
particle_size = 0.1
particle_spacing = 0.05

bounds_size_x = 19.0
bounds_size_y = 10.25

# SPH constants
smoothing_radius = 0.45
particle_mass = 1.0
target_density = 15
pressure_multiplier = 140.0
viscosity_strength = 0.2

max_display_speed = 2.0

# interaction
interaction_radius = 5.0
interaction_strength = 20.0

positions = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
predicted_positions = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
new_velocities = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
densities = ti.field(dtype=ti.f32, shape=num_particles)

render_pos = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
color = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
bounds_verts = ti.Vector.field(2, dtype=ti.f32, shape=8)


@ti.func
def phys_to_canvas(point: ti.template()) -> ti.Vector:
    cx = (point[0] + LOGICAL_SIZE_X * 0.5) / LOGICAL_SIZE_X
    cy = 1.0 - (point[1] + LOGICAL_SIZE_Y * 0.5) / LOGICAL_SIZE_Y
    return ti.Vector([cx, cy])

@ti.kernel
def grid_arrangement():
    particles_per_row = int(ti.sqrt(float(num_particles)))
    particles_per_col = (num_particles - 1) // particles_per_row + 1
    spacing = particle_size * 2.0 + particle_spacing
    for i in range(num_particles):
        col = i % particles_per_row
        row = i // particles_per_row
        x = (col - particles_per_row / 2.0 + 0.5) * spacing
        y = (row - particles_per_col / 2.0 + 0.5) * spacing
        positions[i] = ti.Vector([x, y])
        velocities[i] = ti.Vector([0.0, 0.0])

# taichi  no support for draw.rect as in pygame
@ti.kernel
def init_bounds_verts():
    bx0 = (-bounds_size_x * 0.5 + LOGICAL_SIZE_X * 0.5) / LOGICAL_SIZE_X
    bx1 = ( bounds_size_x * 0.5 + LOGICAL_SIZE_X * 0.5) / LOGICAL_SIZE_X
    by_top    = 1.0 - (-bounds_size_y * 0.5 + LOGICAL_SIZE_Y * 0.5) / LOGICAL_SIZE_Y
    by_bottom = 1.0 - ( bounds_size_y * 0.5 + LOGICAL_SIZE_Y * 0.5) / LOGICAL_SIZE_Y
    # bottom
    bounds_verts[0] = ti.Vector([bx0, by_bottom])
    bounds_verts[1] = ti.Vector([bx1, by_bottom])
    # top 
    bounds_verts[2] = ti.Vector([bx0, by_top])
    bounds_verts[3] = ti.Vector([bx1, by_top])
    # left
    bounds_verts[4] = ti.Vector([bx0, by_bottom])
    bounds_verts[5] = ti.Vector([bx0, by_top])
    # right edge
    bounds_verts[6] = ti.Vector([bx1, by_bottom])
    bounds_verts[7] = ti.Vector([bx1, by_top])


@ti.func
def smoothing_kernel(radius: ti.f32, distance: ti.f32) -> ti.f32:
    volume = (math.pi * (radius ** 4.0)) / 6.0                                                                                                                                                                                                                                                          
    value = ((radius - distance) ** 2.0) / volume
    return value if distance < radius else 0.0 # taichi not support early if return

@ti.func
def smoothing_kernel_derivative(radius: ti.f32, distance: ti.f32) -> ti.f32:
    scale = 12.0 / ((radius ** 4.0) * math.pi)                                                                                                                                                                                                                                                          
    return (distance - radius) * scale if distance < radius else 0.0 # same taichi restriciton

@ti.func
def viscosity_kernel(radius: ti.f32, distance: ti.f32) -> ti.f32:
    scale = 4.0 / (math.pi * radius ** 8)                                                                                                                                                                                                                                                               
    value = radius * radius - distance * distance                                                                                                                                                                                                                                                       
    return value * value * value * scale if distance < radius else 0.0    

@ti.func
def convert_density_to_pressure(density: ti.f32) -> ti.f32:
    return (density - target_density) * pressure_multiplier


@ti.func
def calculate_shared_pressure(density_a: ti.f32, density_b: ti.f32) -> ti.f32:
    return (convert_density_to_pressure(density_a) + convert_density_to_pressure(density_b)) * 0.5


@ti.kernel
def calculate_densities():
    for i in range(num_particles):
        density = 0.0
        for j in range(num_particles):
            # Use predicted positions for distance
            dist = (predicted_positions[i] - predicted_positions[j]).norm()
            influence = smoothing_kernel(smoothing_radius, dist)
            density += particle_mass * influence
        densities[i] = density

@ti.func
def calculate_pressure_force(particle_index: int) -> ti.Vector:
    pressure_force = ti.Vector([0.0, 0.0])
    viscosity_force = ti.Vector([0.0, 0.0])
    density_self = densities[particle_index]

    for other_particle_index in range(num_particles):
        if particle_index == other_particle_index:
            continue

        offset = predicted_positions[other_particle_index] - predicted_positions[particle_index]
        dst = offset.norm()

        dir = ti.Vector([ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0])
        if dst == 0:
            if len(dir) > 0:
                dir = dir.norm()
            else:
                dir = ti.Vector([0.0, 1.0])# pygame.math.Vector2(0, 1)
        else:
            dir = offset / dst

        slope = smoothing_kernel_derivative(smoothing_radius, dst)
        density_other = densities[other_particle_index]

        if density_other == 0:
            continue

        shared_pressure = calculate_shared_pressure(density_self, density_other)
        pressure_force += dir * shared_pressure * slope * particle_mass / density_other

        influence = viscosity_kernel(smoothing_radius, dst)
        viscosity_force += (velocities[other_particle_index] - velocities[particle_index]) * influence

    return pressure_force + viscosity_force * viscosity_strength

@ti.func
def calculate_interaction_force(input_x: ti.f32, input_y: ti.f32, radius: ti.f32, strength: ti.f32, particle_index: int) -> ti.Vector:
    offset = ti.Vector([input_x, input_y]) - positions[particle_index]
    sqr_dst = offset.dot(offset)
    result = ti.Vector([0.0, 0.0])
    if sqr_dst < radius * radius:
        dst = ti.sqrt(sqr_dst)
        dir_to_input = ti.Vector([0.0, 0.0])
        if dst > 1e-6:
            dir_to_input = offset / dst
        centre_t = 1.0 - dst / radius
        result = (dir_to_input * strength - velocities[particle_index]) * centre_t
    return result


@ti.kernel
def update(dt: ti.f32):
    # predict next positions before calculating densities
    prediction_time_step = 1.0 / 120.0  # konstant time step
    for i in range(num_particles):
        velocities[i] += ti.Vector([0.0, gravity]) * dt
        predicted_positions[i] = positions[i] + velocities[i] * prediction_time_step


@ti.kernel
def apply_forces(dt: ti.f32, mouse_x: ti.f32, mouse_y: ti.f32, mouse_strength: ti.f32):
    for i in range(num_particles):
        dv = ti.Vector([0.0, 0.0])
        pressure_force = calculate_pressure_force(i)
        if densities[i] > 0.0:
            dv += (pressure_force / densities[i]) * dt

        if mouse_strength != 0.0:
            interaction = calculate_interaction_force(mouse_x, mouse_y, interaction_radius, mouse_strength, i)
            dv += interaction * dt

        new_velocities[i] = velocities[i] + dv


@ti.kernel
def apply_new_velocities():
    for i in range(num_particles):
        velocities[i] = new_velocities[i]


@ti.kernel
def resolve_collisions(dt: ti.f32):
    half_x = bounds_size_x * 0.5 - particle_size
    half_y = bounds_size_y * 0.5 - particle_size
    for i in range(num_particles):
        positions[i] += velocities[i] * dt
        if ti.abs(positions[i][0]) > half_x:
            positions[i][0] = half_x * (1.0 if positions[i][0] > 0.0 else -1.0)
            velocities[i][0] *= -collision_damping
        if ti.abs(positions[i][1]) > half_y:
            positions[i][1] = half_y * (1.0 if positions[i][1] > 0.0 else -1.0)
            velocities[i][1] *= -collision_damping


@ti.kernel
def update_render_data():
    for i in range(num_particles):
        render_pos[i] = phys_to_canvas(positions[i])
        color[i] = ti.Vector([0, 0, 255])

def main():
    window = ti.ui.Window("SPH GPU", (SCREEN_W, SCREEN_H), fps_limit=60)
    canvas = window.get_canvas()
    gui = window.get_gui()

    grid_arrangement()
    init_bounds_verts()

    particle_radius_norm = particle_size / LOGICAL_SIZE_X

    t_last = time.perf_counter()
    fps = 0.0

    while window.running:
        t_now = time.perf_counter()
        frame_time = t_now - t_last
        t_last = t_now

        cx, cy = window.get_cursor_pos()
        mouse_lx = cx * LOGICAL_SIZE_X - LOGICAL_SIZE_X * 0.5
        mouse_ly = (1.0 - cy) * LOGICAL_SIZE_Y - LOGICAL_SIZE_Y * 0.5

        mouse_strength = 0.0
        if window.is_pressed(ti.ui.LMB):
            mouse_strength = interaction_strength
        elif window.is_pressed(ti.ui.RMB):
            mouse_strength = -interaction_strength

        fixed_dt = 0.004
        for _ in range(4):
            update(fixed_dt)
            calculate_densities()
            apply_forces(fixed_dt, mouse_lx, mouse_ly, mouse_strength)
            apply_new_velocities()
            resolve_collisions(fixed_dt)

        update_render_data()

        canvas.set_background_color((0.0, 0.0, 0.0))
        canvas.lines(bounds_verts, width=0.0015, color=(0.4, 0.4, 0.4))
        canvas.circles(render_pos, radius=particle_radius_norm, per_vertex_color=color)

        fps = 1.0 / frame_time
        with gui.sub_window("", x=0.01, y=0.01, width=0.12, height=0.03):
            gui.text(f"FPS: {fps:.1f}")

        window.show()


if __name__ == "__main__":
    main()
