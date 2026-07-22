"""Lennard-Jones MD temperature sweep with local-density distributions and Binder parameters.

For every target temperature, the existing cell list algorithm for MD
is run and the simulation box is divided into three sets of analysis blocks.
For each sampled configuration, the local block densities rho_i are used to
calculate the local density distribution P(rho), the Binder parameter and surface tension.
"""

from pathlib import Path
import argparse

import numpy as np
from numba import njit
import time


# Lennard-Jones reduced units: sigma = epsilon = mass = k_B = 1.
N = 1000

# Elongated coexistence box: the x-side is doubled.
Lx = 24.0
Ly = 12.0
Lz = 12.0

box = np.array([Lx, Ly, Lz], dtype=np.float64)

rcut = 2.5
dt = 0.005
rcut2 = rcut**2

Ecut = 4.0 * (rcut**-12 - rcut**-6)

# Maximum number of cells such that every cell is at least rcut wide.
ncell_x = max(1, int(Lx / rcut))
ncell_y = max(1, int(Ly / rcut))
ncell_z = max(1, int(Lz / rcut))

# Actual cell dimensions.
cell_x = Lx / ncell_x
cell_y = Ly / ncell_y
cell_z = Lz / ncell_z

# Different block sizes proportional to box size, i.e. 2:1:1 for the local-density/Binder analysis.
block_divisions = np.array([
    [4, 2, 2],
    [6, 3, 3],
    [8, 4, 4],
], dtype=np.int64)

def init_positions(n_particles, box):
    """Place particles in the half of the x direction. """
    occupied_x = 0.5 * box[0]
    target_spacing = 1.05
    nx = max(1, int(occupied_x / target_spacing))
    ny = max(1, int(box[1] / target_spacing))
    nz = max(1, int(box[2] / target_spacing))
    while nx * ny * nz < n_particles:
        nx += 1
    x = (np.arange(nx) + 0.5) * occupied_x / nx
    y = (np.arange(ny) + 0.5) * box[1] / ny
    z = (np.arange(nz) + 0.5) * box[2] / nz
    grid = np.array(np.meshgrid(x, y, z, indexing="ij")).reshape(3, -1).T
    positions = grid[:n_particles].copy()
    # Center the slab in the elongated box.
    positions[:, 0] += 0.25 * box[0]
    return positions


def init_velocities(n_particles, temperature, rng):
    """Draw velocities, remove centre-of-mass motion, and set temperature."""
    # Start with random Cartesian velocity components in [-0.5, 0.5).
    velocities = rng.random((n_particles, 3)) - 0.5
    # Zero total momentum by removing the mean velocity.
    velocities -= velocities.mean(axis=0)
    # Rescale so sum(v^2)/(3N) equals the requested reduced temperature.
    mean_squared_component = np.sum(velocities**2) / (3.0 * n_particles)
    velocities *= np.sqrt(temperature / mean_squared_component)
    return velocities




# Neighbour cells of the current cell, including the cell itself.
neighbor_offsets = (
    (0, 0, 0),

    (1, -1, -1),
    (1, -1,  0),
    (1, -1,  1),
    (1,  0, -1),
    (1,  0,  0),
    (1,  0,  1),
    (1,  1, -1),
    (1,  1,  0),
    (1,  1,  1),

    (0, 1, -1),
    (0, 1,  0),
    (0, 1,  1),

    (0, 0, 1),
)

# Building the cells.
@njit
def build_cells(positions):
    n_particles = positions.shape[0]
    number_of_cells=ncell_x*ncell_y*ncell_z

    number_of_part_in_cell = np.zeros(number_of_cells, dtype=np.int64)
    """-1 so that empty does not equal 1, since for particle with number one it would also be 1"""
    particles_in_cell = -np.ones((number_of_cells, n_particles), dtype=np.int64)

    """loop over all particles"""
    for i in range(n_particles):
        """compute cell index"""
        cx=int(positions[i,0]/cell_x)%ncell_x
        cy=int(positions[i,1]/cell_y)%ncell_y
        cz=int(positions[i,2]/cell_z)%ncell_z

        """compute the cell number of the neighbouring cell"""
        cell_number= cx+ncell_x*(cy+ncell_y*cz)

        """check how many particles are already in this cell, to store particle i in the right place"""
        part=number_of_part_in_cell[cell_number]

        """store the particle number i in a 2D array with first index the cell_number and second the index of how many particles in this cell"""
        particles_in_cell[cell_number,part]=i

        """increase number of particles in this cell"""
        number_of_part_in_cell[cell_number]+=1

    return number_of_part_in_cell, particles_in_cell


# Calculation of the local density in the blocks.
@njit
def sample_block_densities(positions, box, nx, ny, nz):
    """Return the particle density in every analysis block."""
    n_blocks = nx * ny * nz
    counts = np.zeros(n_blocks, dtype=np.int64)

    block_volume = (
        (box[0] / nx)
        * (box[1] / ny)
        * (box[2] / nz)
    )

    for i in range(positions.shape[0]):
        bx = int(positions[i, 0] / box[0] * nx) % nx
        by = int(positions[i, 1] / box[1] * ny) % ny
        bz = int(positions[i, 2] / box[2] * nz) % nz

        block_id = bx + nx * (by + ny * bz)
        counts[block_id] += 1

    return counts.astype(np.float64) / block_volume


# Calculate the second and forth moments of the density distribution for Binder cumulant. 
@njit
def density_moments(densities):
    """Return spatial second and fourth central density moments."""
    mean_density = np.mean(densities)
    m2 = 0.0
    m4 = 0.0

    for rho in densities:
        delta = rho - mean_density
        delta2 = delta * delta
        m2 += delta2
        m4 += delta2 * delta2

    n_blocks = densities.size
    return m2 / n_blocks, m4 / n_blocks


@njit
def compute_forces(positions, box):
    """Return LJ forces, potential energy, and the Eq. (4) from Watanabe surface tension.
    For surface tension was considered, that x-side is doubled and not z-side like in the paper:
    2 gamma Ly Lz = sum_{i<j} [(y_ij^2 + z_ij^2 - 2 x_ij^2)/(2 r_ij)] V'(r_ij).
    """
    # A separate force vector is accumulated for each particle.
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    # Right-hand-side pair sum for the surface tension formula.
    surface_tension_pair_sum = 0.0
    rij = np.empty(3, dtype=np.float64)
    n_particles = positions.shape[0]

    # j starts at i+1 so every particle pair is evaluated exactly once.
    """from the positions build the cell"""
    number_of_particles_in_cell, particles_in_cell = build_cells(positions)

    for i in range(n_particles):
        """compute cell index of particle i"""
        cx=int(positions[i,0]/cell_x)%ncell_x
        cy=int(positions[i,1]/cell_y)%ncell_y
        cz=int(positions[i,2]/cell_z)%ncell_z

        """loop over all possible neighbour directions by looking at shifts dx,dy,dz"""
        for d in neighbor_offsets:
            dx, dy, dz = d
            """compute the neighbour cell index by applying the shifts"""
            ncx=(cx+dx)%ncell_x
            ncy=(cy+dy)%ncell_y
            ncz=(cz+dz)%ncell_z

            """compute the cell number of the neighbouring cell"""
            neighbour_cell_number= ncx+ncell_x*(ncy+ncell_y*ncz)

            """loop over all particles in this neighbouring cell"""
            for part in range(number_of_particles_in_cell[neighbour_cell_number]):
                """fetch the particle number j from this stored particle index"""
                j=particles_in_cell[neighbour_cell_number,part]
                        
            
                # In the same cell, avoid self-interaction and double counting.
                if dx == 0 and dy == 0 and dz == 0:
                    if j <= i:
                        continue

                for k in range(3):
                    # Minimum-image displacement r_i-r_j.
                    rij[k] = positions[i, k] - positions[j, k]
                    rij[k] -= box[k] * np.rint(rij[k] / box[k])

                r2 = rij[0]**2 + rij[1]**2 + rij[2]**2
                # Interactions outside the cutoff are neglected.
                if 0.0 < r2 < rcut2:
                    inv_r2 = 1.0 / r2
                    inv_r6 = inv_r2**3
                    inv_r12 = inv_r6**2
                    force_factor = (48.0 * inv_r12 - 24.0 * inv_r6) * inv_r2
                    # Newton's third law: equal and opposite pair forces.
                    for k in range(3):
                        pair_force = force_factor * rij[k]
                        forces[i, k] += pair_force
                        forces[j, k] -= pair_force

                    # Shift the LJ potential so it goes continuously to zero at rcut.
                    potential_energy += 4.0 * (inv_r12 - inv_r6) - Ecut

                    # Calculation of the V'(r).
                    r = np.sqrt(r2)
                    dV_dr = 24.0 * (inv_r6 - 2.0 * inv_r12) / r

                    # Interface normal x and area Ly*Lz. surface tension as difference of tangential and normal pressure 
                    surface_tension_pair_sum += (
                        (
                            rij[1] * rij[1]
                            + rij[2] * rij[2]
                            - 2.0 * rij[0] * rij[0]
                        )
                        * dV_dr
                        / (2.0 * r)
                    )

    # Periodic boundaries produce two interfaces, hence the factor 2.
    surface_tension = surface_tension_pair_sum / (2.0 * box[1] * box[2])
    return forces, potential_energy, surface_tension


@njit
def calc_positionsv2v(positions, velocities, step_size, box):
    """Drift positions for one full step and wrap them into the box."""
    # Calculate positions at the next integer time step, s = v*t.
    positions += velocities * step_size
    # Periodic boundaries map every coordinate back to [0, box_length).
    positions %= box
    return positions


@njit
def calc_velocitiesv2v(velocities, forces, step_size):
    #calculate velocities from the forces
    velocities += 0.5 * forces * step_size
    return velocities




@njit
def bussi_thermostat(velocities, T_target, step_size, tau_t):
    """Apply Bussi thermostat for equilibration.
    Rescale velocities to achieve the target temperature
    """
    if tau_t <= 0.0:
        return velocities

    n_particles = velocities.shape[0]
    n_dof = 3 * n_particles - 3
    kinetic_energy = 0.5 * np.sum(velocities**2)
    if kinetic_energy <= 0.0 or n_dof <= 0:
        return velocities

    #calculate the target kinetic energy
    target_kinetic_energy = 0.5 * n_dof * T_target
    c = np.exp(-step_size / tau_t)

    #calculate the target kinetic energy using Bussi 
    R = np.random.normal()
    S = np.random.gamma(0.5 * (n_dof - 1), 2.0)

    new_kinetic_energy = (
        c * kinetic_energy
        + (1.0 - c) * target_kinetic_energy * (S + R * R) / n_dof
        + 2.0 * R * np.sqrt(
            c * (1.0 - c) * kinetic_energy * target_kinetic_energy / n_dof
        )
    )

    #make sure it is not negative
    if new_kinetic_energy < 0.0:
        new_kinetic_energy = 0.0

    #calculate factor to rescale the velocites
    alpha = np.sqrt(new_kinetic_energy / kinetic_energy)
    velocities *= alpha
    return velocities


@njit
def simulate(positions, velocities, box, step_size, n_steps,
             sample_every, trajectory_every, T_target, tau_t,
             equilibration_steps, density_every, block_divisions,
             density_min, density_max, density_bins):
    """Run velocity rescale and retain sampled observables and trajectory."""
    # Forces at the initial positions are needed for the first integration step.
    forces, potential_energy, surface_tension = compute_forces(positions,box)
    # Arrays to store the data.
    n_samples = n_steps // sample_every + 1
    n_frames = n_steps // trajectory_every + 1
    measurements = np.empty((n_samples, 9))
    trajectory = np.empty((n_frames, positions.shape[0], 3))
    sample_index = 0
    frame_index = 0

    # Arrays for local-density histograms and Binder parameters.
    n_partitions = block_divisions.shape[0]
    density_histograms = np.zeros((n_partitions, density_bins), dtype=np.int64)
    sum_m2 = np.zeros(n_partitions)
    sum_m4 = np.zeros(n_partitions)
    density_samples = np.zeros(n_partitions, dtype=np.int64)
    density_width = (density_max - density_min) / density_bins

    # Surface tension is sampled after equilibration at the same interval as
    # the local-density analysis.  The first and second moments are retained
    # so the temperature-sweep plot can include a standard-error estimate.
    surface_tension_sum = 0.0
    surface_tension_sum2 = 0.0
    surface_tension_samples = 0

    for step in range(n_steps + 1):
        # Thermodynamic observables need not be stored at every integration step.
        if step % sample_every == 0:
            kinetic_energy = 0.5 * np.sum(velocities**2)
            # Equipartition in 3D gives T = 2 K / (3 N), with k_B = 1.
            temperature = 2.0 * kinetic_energy / (3.0 * positions.shape[0] - 3.0)
            # Momentum should remain approximately zero throughout the run.
            momentum = np.sum(velocities, axis=0)
            measurements[sample_index, 0] = step
            measurements[sample_index, 1] = step * step_size
            measurements[sample_index, 2] = temperature
            measurements[sample_index, 3] = kinetic_energy + potential_energy
            measurements[sample_index, 4] = kinetic_energy
            measurements[sample_index, 5] = potential_energy
            measurements[sample_index, 6:9] = momentum
            sample_index += 1
        if step % trajectory_every == 0:
            # Store fewer XYZ frames than integration steps to limit file size.
            trajectory[frame_index] = positions
            frame_index += 1

        # Local-density statistics and surface tension are collected only
        # after equilibration, i.e. the thermostat is switched off.
        if step >= equilibration_steps and step % density_every == 0:
            surface_tension_sum += surface_tension
            surface_tension_sum2 += surface_tension * surface_tension
            surface_tension_samples += 1

            for p in range(n_partitions):
                nx = block_divisions[p, 0]
                ny = block_divisions[p, 1]
                nz = block_divisions[p, 2]

                local_densities = sample_block_densities(
                    positions, box, nx, ny, nz
                )
                m2, m4 = density_moments(local_densities)

                sum_m2[p] += m2
                sum_m4[p] += m4
                density_samples[p] += 1

                for rho in local_densities:
                    bin_index = int((rho - density_min) / density_width)
                    if 0 <= bin_index < density_bins:
                        density_histograms[p, bin_index] += 1

        if step == n_steps:
            break

        # Velocity, position and force are updated.
        velocities = calc_velocitiesv2v(velocities, forces, step_size)
        positions = calc_positionsv2v(positions, velocities, step_size,
                                      box)
        forces, potential_energy, surface_tension = compute_forces(positions, box)
        velocities = calc_velocitiesv2v(velocities, forces, step_size)

        # If the system is still equilibrating, apply the Bussi thermostat to rescale the velocities.
        if step < equilibration_steps:
            velocities = bussi_thermostat(
                velocities, T_target, step_size, tau_t
            )

    return (
        positions, velocities, measurements, trajectory,
        density_histograms, sum_m2, sum_m4, density_samples,
        surface_tension_sum, surface_tension_sum2, surface_tension_samples
    )






def write_xyz(filename, trajectory, trajectory_every, step_size, element="Ar"):
    """Write a complete (overwriting) wrapped XYZ trajectory."""
    # Opening with mode "w" prevents a new run being appended to an old one.
    with open(filename, "w", encoding="utf-8") as handle:
        for frame_index, positions in enumerate(trajectory):
            # XYZ frames begin with particle count followed by one comment line.
            handle.write(f"{positions.shape[0]}\n")
            time = frame_index * trajectory_every * step_size
            handle.write(f"time = {time:.6f}; box = {Lx:.6f} {Ly:.6f} {Lz:.6f}\n")
            for x, y, z in positions:
                handle.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")


def main():
    # Start the timer before argument parsing so the reported runtime includes
    # setup, Numba compilation, simulation, file writing, and plotting.
    start_time = time.perf_counter()

    # Command-line options allow shorter tests without editing the source code.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--trajectory-every", type=int, default=500)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.74, 0.80, 0.86, 0.90, 0.94, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20]
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--thermostat-tau",
        type=float,
        default=1.0,
        help="Bussi thermostat relaxation time; <= 0 disables it"
    )
    parser.add_argument(
        "--equilibration-steps",
        type=int,
        default=100_000,
        help="Steps discarded before local-density sampling"
    )
    parser.add_argument(
        "--density-every",
        type=int,
        default=100,
        help="Interval between local-density samples"
    )
    parser.add_argument("--density-min", type=float, default=0.0)
    parser.add_argument("--density-max", type=float, default=1.5)
    parser.add_argument("--density-bins", type=int, default=100)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()

    if args.steps < 0:
        raise ValueError("--steps must be non-negative")

    if args.sample_every <= 0:
        raise ValueError("--sample-every must be a positive integer")

    if args.trajectory_every <= 0:
        raise ValueError("--trajectory-every must be a positive integer")

    if args.equilibration_steps < 0 or args.equilibration_steps > args.steps:
        raise ValueError("--equilibration-steps must lie between 0 and --steps")

    if args.density_every <= 0:
        raise ValueError("--density-every must be a positive integer")

    if args.density_bins <= 0:
        raise ValueError("--density-bins must be a positive integer")

    if args.density_max <= args.density_min:
        raise ValueError("--density-max must be larger than --density-min")

    args.results.mkdir(parents=True, exist_ok=True)

    all_measurements = {}
    all_density_histograms = {}
    all_binder = {}
    all_surface_tension = {}
    header = (
        "step time temperature total_energy kinetic_energy "
        "potential_energy Px Py Pz"
    )

    for run_index, temperature in enumerate(args.temperatures):
        print(
            f"Running T_target={temperature:g} "
            f"for {args.steps} steps ...",
            flush=True
        )

        # A reproducible but different random velocity set is used for each run.
        rng = np.random.default_rng(args.seed + run_index)

        # C-contiguous float arrays give Numba efficient memory access.
        positions = np.ascontiguousarray(
            init_positions(N, box),
            dtype=np.float64
        )
        velocities = np.ascontiguousarray(
            init_velocities(N, temperature, rng),
            dtype=np.float64
        )

        (
            positions, velocities, measurements, trajectory,
            density_histograms, sum_m2, sum_m4, density_samples,
            surface_tension_sum, surface_tension_sum2,
            surface_tension_samples
        ) = simulate(
            positions,
            velocities,
            box,
            dt,
            args.steps,
            args.sample_every,
            args.trajectory_every,
            temperature,
            args.thermostat_tau,
            args.equilibration_steps,
            args.density_every,
            block_divisions,
            args.density_min,
            args.density_max,
            args.density_bins
        )

        tag = f"T{temperature:g}".replace(".", "p")

        # Save numerical data separately from the figures for later analysis.
        np.savetxt(
            args.results / f"measurements_{tag}.txt",
            measurements,
            header=header
        )

        write_xyz(
            args.results / f"trajectory_{tag}.xyz",
            trajectory,
            args.trajectory_every,
            dt
        )

        all_measurements[temperature] = measurements
        all_density_histograms[temperature] = density_histograms

        mean_m2 = np.full(block_divisions.shape[0], np.nan)
        mean_m4 = np.full(block_divisions.shape[0], np.nan)
        valid = density_samples > 0
        mean_m2[valid] = sum_m2[valid] / density_samples[valid]
        mean_m4[valid] = sum_m4[valid] / density_samples[valid]

        binder = np.full(block_divisions.shape[0], np.nan)
        nonzero = valid & (mean_m2 > 0.0)
        binder[nonzero] = mean_m4[nonzero] / mean_m2[nonzero]**2
        all_binder[temperature] = binder

        if surface_tension_samples > 0:
            gamma_mean = surface_tension_sum / surface_tension_samples

            if surface_tension_samples > 1:
                gamma_variance = (
                    surface_tension_sum2
                    - surface_tension_samples * gamma_mean**2
                ) / (surface_tension_samples - 1)
                gamma_variance = max(gamma_variance, 0.0)
                gamma_error = np.sqrt(
                    gamma_variance / surface_tension_samples
                )
            else:
                gamma_error = np.nan
        else:
            gamma_mean = np.nan
            gamma_error = np.nan

        all_surface_tension[temperature] = (
            gamma_mean,
            gamma_error,
        )

        print(
            f"  surface tension gamma={gamma_mean:.6e} "
            f"+/- {gamma_error:.3e}",
            flush=True,
        )

        density_edges = np.linspace(
            args.density_min, args.density_max, args.density_bins + 1
        )
        density_centers = 0.5 * (density_edges[:-1] + density_edges[1:])
        np.savetxt(
            args.results / f"local_density_hist_{tag}.txt",
            np.column_stack((density_centers, density_histograms.T)),
            header=(
                "rho "
                + " ".join(
                    f"counts_{nx}x{ny}x{nz}"
                    for nx, ny, nz in block_divisions
                )
            ),
        )

        drift = np.max(
            np.abs(measurements[:, 3] - measurements[0, 3])
        )
        max_momentum = np.max(
            np.linalg.norm(measurements[:, 6:9], axis=1)
        )

        print(
            f"  max |E-E0|={drift:.3e}; "
            f"max |P|={max_momentum:.3e}",
            flush=True
        )

    # Save temperature-sweep summary data for the separate plotting notebook.
    temperatures = np.array(sorted(all_binder), dtype=float)
    binder_values = np.array([all_binder[T] for T in temperatures])
    binder_header = "temperature " + " ".join(
        f"U_{nx}x{ny}x{nz}" for nx, ny, nz in block_divisions
    )
    np.savetxt(
        args.results / "binder_parameter_vs_temperature.txt",
        np.column_stack((temperatures, binder_values)),
        header=binder_header,
    )

    gamma = np.array(
        [all_surface_tension[T][0] for T in temperatures], dtype=float
    )
    gamma_error = np.array(
        [all_surface_tension[T][1] for T in temperatures], dtype=float
    )
    np.savetxt(
        args.results / "surface_tension_vs_temperature.txt",
        np.column_stack((temperatures, gamma, gamma_error)),
        header="temperature gamma gamma_standard_error",
    )

    # Store simulation and analysis settings in one machine-readable file.
    # The plotting notebook loads this file first and then discovers all
    # temperature-dependent text files in the same results directory.
    np.savez(
        args.results / "simulation_metadata.npz",
        temperatures=temperatures,
        block_divisions=block_divisions,
        density_min=float(args.density_min),
        density_max=float(args.density_max),
        density_bins=int(args.density_bins),
        N=int(N),
        box=box,
        dt=float(dt),
        steps=int(args.steps),
        sample_every=int(args.sample_every),
        trajectory_every=int(args.trajectory_every),
        equilibration_steps=int(args.equilibration_steps),
        density_every=int(args.density_every),
        thermostat_tau=float(args.thermostat_tau),
        seed=int(args.seed),
    )

    print(
        f"Results written to {args.results.resolve()}",
        flush=True
    )

    # Stop the timer only after all simulations, saving, and plotting are done.
    runtime = time.perf_counter() - start_time

    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = runtime % 60

    print(
        f"Total runtime: "
        f"{hours:02d}:{minutes:02d}:{seconds:06.3f} "
        f"({runtime:.3f} s)",
        flush=True
    )


if __name__ == "__main__":
    main()