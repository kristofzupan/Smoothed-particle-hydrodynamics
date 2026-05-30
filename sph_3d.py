import taichi as ti
import numpy as np
import math
import time

ti.init(arch=ti.metal)

SCREEN_W = 1920
SCREEN_H = 1080

num_particles = 8192
gravity = 10
collision_damping = 0.95
particle_size = 0.1
particle_spacing = 0.15

bounds_size_x = 15.0
bounds_size_y = 15.0
bounds_size_z = 10.0

# SPH constants
smoothing_radius = 0.7
particle_mass = 1.0
target_density = 26.0
pressure_multiplier = 500.0
viscosity_strength = 0.05

positions = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
predicted_positions = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
new_velocities = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
densities = ti.field(dtype=ti.f32, shape=num_particles)

bounds_verts = ti.Vector.field(3, dtype=ti.f32, shape=24)

@ti.kernel
def grid_arrangement():
    n_side = int(ti.pow(float(num_particles), 1.0 / 3.0)) + 1
    spacing = particle_size * 2.0 + particle_spacing
    for i in range(num_particles):
        ix = i % n_side
        iy = (i // n_side) % n_side
        iz = i // (n_side * n_side)
        x = (float(ix) - float(n_side) * 0.5 + 0.5) * spacing
        y = (float(iy) - float(n_side) * 0.5 + 0.5) * spacing
        z = (float(iz) - float(n_side) * 0.5 + 0.5) * spacing
        positions[i] = ti.Vector([x, y, z])
        velocities[i] = ti.Vector([0.0, 0.0, 0.0])

# taichi no support for draw.rect, build wireframe manually
def init_bounds_verts():
    w, h, d = bounds_size_x * 0.5, bounds_size_y * 0.5, bounds_size_z * 0.5
    corners = np.array([
        [-w,-h,-d], [ w,-h,-d], [ w, h,-d], [-w, h,-d],
        [-w,-h, d], [ w,-h, d], [ w, h, d], [-w, h, d],
    ], dtype=np.float32)
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    verts = np.array([[corners[a], corners[b]] for a, b in edges], dtype=np.float32).reshape(-1, 3)
    bounds_verts.from_numpy(verts)


# updated for 3d
@ti.func
def smoothing_kernel(radius: ti.f32, distance: ti.f32) -> ti.f32:
    volume = 2.0 * math.pi * radius ** 5 / 15.0 
    value = ((radius - distance) ** 2.0) / volume
    return value if distance < radius else 0.0 # taichi not support early if return

@ti.func
def smoothing_kernel_derivative(radius: ti.f32, distance: ti.f32) -> ti.f32:
    scale = 15.0 / (math.pi * radius ** 5)  
    return (distance - radius) * scale if distance < radius else 0.0 # same taichi restriciton

@ti.func
def viscosity_kernel(radius: ti.f32, distance: ti.f32) -> ti.f32:
    scale = 315.0 / (64.0 * math.pi * radius ** 9)  
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
    pressure_force = ti.Vector([0.0, 0.0, 0.0])
    density_self = densities[particle_index]

    for other_particle_index in range(num_particles):
        if particle_index == other_particle_index:
            continue

        offset = predicted_positions[other_particle_index] - predicted_positions[particle_index]
        dst = offset.norm()

        dir = ti.Vector([ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0, ti.random() * 2.0 - 1.0])
        if dst == 0:
            dir = dir.normalized()
        else:
            dir = offset / dst

        slope = smoothing_kernel_derivative(smoothing_radius, dst)
        density_other = densities[other_particle_index]

        if density_other == 0:
            continue

        shared_pressure = calculate_shared_pressure(density_self, density_other)
        pressure_force += dir * shared_pressure * slope * particle_mass / density_other

    return pressure_force


@ti.kernel
def update(dt: ti.f32):
    # predict next pos before calc dens
    prediction_time_step = 1.0 / 120.0  # konstant time step
    for i in range(num_particles):
        velocities[i] += ti.Vector([0.0, -gravity, 0.0]) * dt
        predicted_positions[i] = positions[i] + velocities[i] * prediction_time_step


@ti.kernel
def apply_forces(dt: ti.f32):
    for i in range(num_particles):
        dv = ti.Vector([0.0, 0.0, 0.0])
        pressure_force = calculate_pressure_force(i)
        if densities[i] > 0.0:
            dv += (pressure_force / densities[i]) * dt

        new_velocities[i] = velocities[i] + dv


@ti.kernel
def apply_new_velocities():
    for i in range(num_particles):
        velocities[i] = new_velocities[i]


@ti.kernel
def apply_viscosity(dt: ti.f32):
    for i in range(num_particles):
        visc_force = ti.Vector([0.0, 0.0, 0.0])
        for j in range(num_particles):
            if i == j:
                continue
            offset = predicted_positions[j] - predicted_positions[i]
            dst = offset.norm()
            influence = viscosity_kernel(smoothing_radius, dst)
            visc_force += (velocities[j] - velocities[i]) * influence
        velocities[i] += visc_force * viscosity_strength * dt


@ti.kernel
def resolve_collisions(dt: ti.f32):
    half_x = bounds_size_x * 0.5 - particle_size
    half_y = bounds_size_y * 0.5 - particle_size
    half_z = bounds_size_z * 0.5 - particle_size
    for i in range(num_particles):
        positions[i] += velocities[i] * dt
        if ti.abs(positions[i][0]) > half_x:
            positions[i][0] = half_x * (1.0 if positions[i][0] > 0.0 else -1.0)
            velocities[i][0] *= -collision_damping
        if ti.abs(positions[i][1]) > half_y:
            positions[i][1] = half_y * (1.0 if positions[i][1] > 0.0 else -1.0)
            velocities[i][1] *= -collision_damping
        if ti.abs(positions[i][2]) > half_z:
            positions[i][2] = half_z * (1.0 if positions[i][2] > 0.0 else -1.0)
            velocities[i][2] *= -collision_damping

def main():
    window = ti.ui.Window("SPH", (SCREEN_W, SCREEN_H), fps_limit=30)
    canvas = window.get_canvas()
    scene = window.get_scene()
    gui = window.get_gui()

    camera = ti.ui.Camera()
    camera.fov(45)

    grid_arrangement()
    init_bounds_verts()

    # orbit camera
    orbit_yaw = 0.4
    orbit_pitch = 0.3
    orbit_dist = bounds_size_x * 2.5
    prev_cursor = (0.0, 0.0)
    was_lmb = False

    t_last = time.perf_counter()
    fps = 0.0

    while window.running:
        t_now = time.perf_counter()
        frame_time = t_now - t_last
        t_last = t_now

        cx, cy = window.get_cursor_pos()
        lmb = window.is_pressed(ti.ui.LMB)
        if lmb and was_lmb:
            dx = cx - prev_cursor[0]
            dy = cy - prev_cursor[1]
            orbit_yaw  += dx * 3.5
            orbit_pitch = max(-math.pi * 0.48, min(math.pi * 0.48, orbit_pitch + dy * 2.5))
        was_lmb = lmb
        prev_cursor = (cx, cy)

        if window.is_pressed(ti.ui.UP):
            orbit_dist = max(bounds_size_x * 0.6, orbit_dist - bounds_size_x * 1.5 * frame_time)
        if window.is_pressed(ti.ui.DOWN):
            orbit_dist = min(bounds_size_x * 8.0, orbit_dist + bounds_size_x * 1.5 * frame_time)

        cam_x = orbit_dist * math.cos(orbit_pitch) * math.sin(orbit_yaw)
        cam_y = orbit_dist * math.sin(orbit_pitch)
        cam_z = orbit_dist * math.cos(orbit_pitch) * math.cos(orbit_yaw)

        camera.position(cam_x, cam_y, cam_z)
        camera.lookat(0.0, 0.0, 0.0)
        camera.up(0.0, 1.0, 0.0)


        fixed_dt = 0.008
        for _ in range(5):
            update(fixed_dt)
            calculate_densities()
            apply_forces(fixed_dt)
            apply_new_velocities()
            apply_viscosity(fixed_dt)
            resolve_collisions(fixed_dt)

        scene.set_camera(camera)
        scene.ambient_light((0.3, 0.3, 0.3))
        scene.point_light(pos=(0.0, bounds_size_y * 2.0, bounds_size_z), color=(1.0, 1.0, 1.0))
        scene.particles(positions, radius=particle_size, color=(0.0,0.0,1.0))
        scene.lines(bounds_verts, width=1.0, color=(0.4, 0.4, 0.4))

        canvas.set_background_color((0.0, 0.0, 0.0))
        canvas.scene(scene)

        fps = 1.0 / frame_time
        with gui.sub_window("", x=0.01, y=0.01, width=0.18, height=0.05):
            gui.text(f"FPS: {fps:.1f}")
            gui.text(f"mouse drag: orbit")
            gui.text(f"Up/Down: zoom")

        window.show()


if __name__ == "__main__":
    main()
