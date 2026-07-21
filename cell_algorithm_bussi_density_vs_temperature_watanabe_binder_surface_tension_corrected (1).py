"""Lennard-Jones MD temperature sweep with local-density distributions and
Watanabe block-density Binder parameters.

For every target temperature, the existing velocity-Verlet/cell-list/Bussi MD
is run and the simulation box is divided into three sets of analysis blocks.
For each sampled configuration, the local block densities rho_i are used to
calculate

    rho_bar = (1/N_b) sum_i rho_i
    m2      = (1/N_b) sum_i (rho_i - rho_bar)^2
    m4      = (1/N_b) sum_i (rho_i - rho_bar)^4
    U       = <m4> / <m2>^2

The script writes a local-density-distribution heat map versus temperature and
plots U(T) for three block sizes.
"""

from pathlib import Path
import argparse

import matplotlib
# Use a non-interactive backend so figures can also be created without a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit
import time


# Lennard-Jones reduced units: sigma = epsilon = mass = k_B = 1.
# Consequently, all lengths, energies, times, and temperatures below are
# dimensionless reduced quantities.
N = 1000

# Elongated coexistence box: the interface normal is the x direction.
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

# Spatial subdivisions used only for the local-density/Binder analysis.
# For a cubic box these make cubic analysis blocks.  If the simulation box
# is elongated, choose divisions proportional to the box lengths, e.g.
# (4, 2, 2), (6, 3, 3), and (8, 4, 4) when Lx = 2 * Ly = 2 * Lz.
block_divisions = np.array([
    [4, 2, 2],
    [6, 3, 3],
    [8, 4, 4],
], dtype=np.int64)

def init_positions(n_particles, box):
    """Place particles in a dense slab occupying half of the x direction.

    This creates the gas-liquid coexistence geometry with two interfaces
    normal to x under periodic boundary conditions.
    """
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
    # Zero total momentum by removing the centre-of-mass velocity.
    velocities -= velocities.mean(axis=0)
    # Rescale so sum(v^2)/(3N) equals the requested reduced temperature.
    mean_squared_component = np.sum(velocities**2) / (3.0 * n_particles)
    velocities *= np.sqrt(temperature / mean_squared_component)
    return velocities





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


@njit
def sample_block_densities(positions, box, nx, ny, nz):
    """Return the instantaneous particle density in every analysis block."""
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
    """Return LJ forces, potential energy, and the Eq. (4) surface tension.

    The elongated direction is x, so the two planar interfaces are normal
    to x and each interface has area Ly * Lz.
    """
    # A separate force vector is accumulated for each particle.
    forces = np.zeros_like(positions)
    potential_energy = 0.0

    # Right-hand-side pair sum of Watanabe et al., Eq. (4), after exchanging
    # x and z because x, rather than z, is the interface-normal direction:
    #
    # 2 gamma Ly Lz =
    # sum_{i<j} [(y_ij^2 + z_ij^2 - 2 x_ij^2)/(2 r_ij)] V'(r_ij).
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
                    # Powers of 1/r^2 avoid repeated, expensive square roots.
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

                    # Explicit implementation of Eq. (4) in Watanabe et al.
                    # The constant energy shift does not change V'(r).
                    r = np.sqrt(r2)
                    dV_dr = 24.0 * (inv_r6 - 2.0 * inv_r12) / r

                    # Paper: interface normal z and area Lx*Ly.
                    # Here:  interface normal x and area Ly*Lz.
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
    # After the first half-kick, these velocities represent the half time step.
    positions += velocities * step_size
    # Periodic boundaries map every coordinate back to [0, box_length).
    positions %= box
    return positions


@njit
def calc_velocitiesv2v(velocities, forces, step_size):
    """Apply one half-kick (unit particle mass)."""
    # With mass equal to one, acceleration is numerically equal to force.
    velocities += 0.5 * forces * step_size
    return velocities




@njit
def bussi_thermostat(velocities, T_target, step_size, tau_t):
    """Apply stochastic velocity rescaling (Bussi thermostat).

    The centre-of-mass momentum is unchanged because every velocity is
    multiplied by the same scalar. Reduced units use k_B = m = 1.
    """
    if tau_t <= 0.0:
        return velocities

    n_particles = velocities.shape[0]
    n_dof = 3 * n_particles - 3
    kinetic_energy = 0.5 * np.sum(velocities**2)
    if kinetic_energy <= 0.0 or n_dof <= 0:
        return velocities

    target_kinetic_energy = 0.5 * n_dof * T_target
    c = np.exp(-step_size / tau_t)

    # R is a standard normal variate and S is chi-square distributed with
    # n_dof - 1 degrees of freedom.  np.random.gamma(k/2, 2) is chi-square(k).
    R = np.random.normal()
    S = np.random.gamma(0.5 * (n_dof - 1), 2.0)

    new_kinetic_energy = (
        c * kinetic_energy
        + (1.0 - c) * target_kinetic_energy * (S + R * R) / n_dof
        + 2.0 * R * np.sqrt(
            c * (1.0 - c) * kinetic_energy * target_kinetic_energy / n_dof
        )
    )

    # Roundoff can only make this very slightly negative.
    if new_kinetic_energy < 0.0:
        new_kinetic_energy = 0.0

    alpha = np.sqrt(new_kinetic_energy / kinetic_energy)
    velocities *= alpha
    return velocities


@njit
def simulate(positions, velocities, box, step_size, n_steps,
             sample_every, trajectory_every, T_target, tau_t,
             equilibration_steps, density_every, block_divisions,
             density_min, density_max, density_bins):
    """Run velocity Verlet and retain sampled observables and trajectory."""
    # Forces at the initial positions are needed for the first half-kick.
    forces, potential_energy, surface_tension = compute_forces(positions,box)
    # Preallocate output arrays because allocation inside a JIT loop is costly.
    n_samples = n_steps // sample_every + 1
    n_frames = n_steps // trajectory_every + 1
    measurements = np.empty((n_samples, 9))
    trajectory = np.empty((n_frames, positions.shape[0], 3))
    sample_index = 0
    frame_index = 0

    # Accumulators for local-density histograms and Binder parameters.
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
        # after equilibration.
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

        # Velocity Verlet sequence: half-kick, full drift, force update,
        # and a second half-kick using the force at the new positions.
        velocities = calc_velocitiesv2v(velocities, forces, step_size)
        positions = calc_positionsv2v(positions, velocities, step_size,
                                      box)
        forces, potential_energy, surface_tension = compute_forces(positions, box)
        velocities = calc_velocitiesv2v(velocities, forces, step_size)

        # Couple the system to the canonical heat bath after a complete
        # velocity-Verlet step, when velocities are defined at an integer time.
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


def plot_measurements(all_measurements, output_dir):
    # Fixed colours make the same initial temperature recognizable in all plots.
    colors = {0.1: "tab:blue", 1.5: "tab:orange"}

    # The first figure directly tests NVE energy and momentum conservation.
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for temperature, values in all_measurements.items():
        label = rf"$T_0={temperature:g}$"
        color = colors.get(temperature)
        energy0 = values[0, 3]
        # Plot the energy change rather than its large absolute value.
        axes[0].plot(values[:, 1], values[:, 3] - energy0,
                     label=label, color=color, linewidth=1)
        momentum_norm = np.linalg.norm(values[:, 6:9], axis=1)
        axes[1].semilogy(values[:, 1], np.maximum(momentum_norm, 1e-18),
                        label=label, color=color, linewidth=1)
    axes[0].set_ylabel(r"$E(t)-E(0)$")
    axes[0].set_title("Conservation checks")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("time")
    axes[1].set_ylabel(r"$|\mathbf{P}|$")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "conservation.png", dpi=180)
    plt.close(fig)

    # The second figure reveals heating/cooling and changes in particle binding.
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for temperature, values in all_measurements.items():
        label = rf"$T_0={temperature:g}$"
        color = colors.get(temperature)
        axes[0].plot(values[:, 1], values[:, 2], label=label,
                     color=color, linewidth=1)
        axes[1].plot(values[:, 1], values[:, 5] / N, label=label,
                     color=color, linewidth=1)
    axes[0].set_ylabel("temperature")
    axes[0].set_title("Thermodynamic time series")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("potential energy / particle")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "thermodynamics.png", dpi=180)
    plt.close(fig)



def plot_local_density_histograms(all_histograms, density_min, density_max,
                                  output_dir, partition_index=1):
    """Plot the normalized local-density distribution for every temperature."""
    if not all_histograms:
        return

    first_histogram = next(iter(all_histograms.values()))
    n_bins = first_histogram.shape[1]
    edges = np.linspace(density_min, density_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = edges[1] - edges[0]

    partition_index = min(partition_index, first_histogram.shape[0] - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for temperature in sorted(all_histograms):
        counts = all_histograms[temperature][partition_index].astype(float)
        normalization = np.sum(counts) * bin_width
        probability_density = counts / normalization if normalization > 0.0 else counts
        ax.plot(centers, probability_density, label=rf"$T={temperature:g}$")

    nx, ny, nz = block_divisions[partition_index]
    ax.set_xlabel(r"local density $\rho_{\mathrm{loc}}$")
    ax.set_ylabel(r"$P(\rho_{\mathrm{loc}})$")
    ax.set_title(f"Local-density distributions ({nx} x {ny} x {nz} blocks)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "local_density_histograms.png", dpi=180)
    plt.close(fig)



def plot_local_density_vs_temperature(all_histograms, density_min, density_max,
                                      output_dir, partition_index=1):
    """Plot P(rho_local | T) as a temperature-density heat map.

    Each row is the normalized local-density distribution at one target
    temperature. This is the direct local-density-distribution-versus-
    temperature plot requested in the project.
    """
    if not all_histograms:
        return

    temperatures = np.array(sorted(all_histograms), dtype=float)
    first_histogram = next(iter(all_histograms.values()))
    n_bins = first_histogram.shape[1]
    partition_index = min(partition_index, first_histogram.shape[0] - 1)

    edges = np.linspace(density_min, density_max, n_bins + 1)
    bin_width = edges[1] - edges[0]

    probability = np.zeros((temperatures.size, n_bins), dtype=float)
    for i, temperature in enumerate(temperatures):
        counts = all_histograms[temperature][partition_index].astype(float)
        normalization = np.sum(counts) * bin_width
        if normalization > 0.0:
            probability[i] = counts / normalization

    # Build temperature-bin edges so each simulated temperature is centred
    # on its heat-map row, including nonuniform temperature sweeps.
    if temperatures.size == 1:
        temperature_edges = np.array([temperatures[0] - 0.5,
                                      temperatures[0] + 0.5])
    else:
        temperature_edges = np.empty(temperatures.size + 1)
        temperature_edges[1:-1] = 0.5 * (temperatures[:-1] + temperatures[1:])
        temperature_edges[0] = temperatures[0] - 0.5 * (temperatures[1] - temperatures[0])
        temperature_edges[-1] = temperatures[-1] + 0.5 * (temperatures[-1] - temperatures[-2])

    fig, ax = plt.subplots(figsize=(8, 6))
    mesh = ax.pcolormesh(edges, temperature_edges, probability, shading="auto")
    fig.colorbar(mesh, ax=ax, label=r"$P(\rho_{\mathrm{loc}}|T)$")

    nx, ny, nz = block_divisions[partition_index]
    ax.set_xlabel(r"local density $\rho_{\mathrm{loc}}$")
    ax.set_ylabel("temperature")
    ax.set_title(
        f"Local-density distribution vs temperature "
        f"({nx} x {ny} x {nz} blocks)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "local_density_vs_temperature.png", dpi=180)
    plt.close(fig)

    # Save the complete temperature-density probability matrix.
    table = np.column_stack((temperatures, probability))
    rho_centers = 0.5 * (edges[:-1] + edges[1:])
    header = "temperature " + " ".join(
        f"P_rho_{rho:.8g}" for rho in rho_centers
    )
    np.savetxt(output_dir / "local_density_vs_temperature.txt",
               table, header=header)



def plot_phase_boundary(all_histograms, density_min, density_max,
                        output_dir, partition_index=1):
    """Estimate gas and liquid coexistence densities from the two histogram peaks.

    Below the critical point the local-density distribution is bimodal.  The
    lower-density maximum estimates rho_g and the higher-density maximum
    estimates rho_l.  Temperatures without two resolved peaks are omitted.
    """
    if not all_histograms:
        return

    first_histogram = next(iter(all_histograms.values()))
    partition_index = min(partition_index, first_histogram.shape[0] - 1)
    n_bins = first_histogram.shape[1]
    edges = np.linspace(density_min, density_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    temperatures_out = []
    gas_densities = []
    liquid_densities = []

    for temperature in sorted(all_histograms):
        counts = all_histograms[temperature][partition_index].astype(float)
        if np.sum(counts) == 0.0:
            continue

        # Light smoothing suppresses single-bin counting noise without moving
        # the broad gas and liquid maxima appreciably.
        smooth = np.convolve(counts, np.ones(5) / 5.0, mode="same")
        mean_density = N / (Lx * Ly * Lz)
        split = np.searchsorted(centers, mean_density)
        if split < 3 or split > n_bins - 3:
            continue

        gas_index = int(np.argmax(smooth[:split]))
        liquid_index = split + int(np.argmax(smooth[split:]))

        # Require a genuine separation between the two maxima.
        if centers[liquid_index] - centers[gas_index] < 3.0 * (edges[1] - edges[0]):
            continue

        temperatures_out.append(temperature)
        gas_densities.append(centers[gas_index])
        liquid_densities.append(centers[liquid_index])

    if not temperatures_out:
        return

    temperatures_out = np.asarray(temperatures_out)
    gas_densities = np.asarray(gas_densities)
    liquid_densities = np.asarray(liquid_densities)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gas_densities, temperatures_out, "o-", label="gas")
    ax.plot(liquid_densities, temperatures_out, "o-", label="liquid")
    diameter = 0.5 * (gas_densities + liquid_densities)
    if temperatures_out.size >= 2:
        coefficients = np.polyfit(temperatures_out, diameter, 1)
        temperature_line = np.linspace(temperatures_out.min(),
                                       temperatures_out.max(), 100)
        density_line = np.polyval(coefficients, temperature_line)
        ax.plot(density_line, temperature_line, "--",
                label="rectilinear diameter")
    ax.set_xlabel(r"density $\rho$")
    ax.set_ylabel("temperature")
    ax.set_title("Gas-liquid coexistence curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "phase_boundary_fig4.png", dpi=180)
    plt.close(fig)

    np.savetxt(
        output_dir / "phase_boundary_fig4.txt",
        np.column_stack((temperatures_out, gas_densities, liquid_densities)),
        header="temperature rho_g rho_l",
    )


def estimate_binder_crossings(temperatures, binder_values):
    """Find pairwise crossings by linear interpolation between temperatures."""
    crossings = []
    n_curves = binder_values.shape[1]

    for a in range(n_curves):
        for b in range(a + 1, n_curves):
            difference = binder_values[:, a] - binder_values[:, b]

            for i in range(len(temperatures) - 1):
                d0 = difference[i]
                d1 = difference[i + 1]

                if not np.isfinite(d0) or not np.isfinite(d1):
                    continue

                if d0 == 0.0:
                    crossings.append(temperatures[i])
                elif d0 * d1 < 0.0:
                    t0 = temperatures[i]
                    t1 = temperatures[i + 1]
                    t_cross = t0 - d0 * (t1 - t0) / (d1 - d0)
                    crossings.append(t_cross)

    return np.array(crossings)


def plot_binder(all_binder, output_dir):
    """Plot Binder parameters and report their approximate crossing."""
    if not all_binder:
        return None

    temperatures = np.array(sorted(all_binder), dtype=float)
    binder_values = np.array([all_binder[T] for T in temperatures])

    fig, ax = plt.subplots(figsize=(8, 5))
    for p in range(block_divisions.shape[0]):
        nx, ny, nz = block_divisions[p]
        ax.plot(
            temperatures,
            binder_values[:, p],
            "o-",
            label=f"{nx} x {ny} x {nz} blocks",
        )

    crossings = estimate_binder_crossings(temperatures, binder_values)
    critical_temperature = None
    if crossings.size > 0:
        critical_temperature = float(np.mean(crossings))
        ax.axvline(
            critical_temperature,
            linestyle="--",
            label=rf"estimated $T_c={critical_temperature:.4f}$",
        )

    ax.set_xlabel("temperature")
    ax.set_ylabel(r"$U=\langle m_4\rangle/\langle m_2\rangle^2$")
    ax.set_title("Watanabe block-density Binder parameter")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "binder_parameter_vs_temperature.png", dpi=180)
    plt.close(fig)

    header = "temperature " + " ".join(
        f"U_{nx}x{ny}x{nz}" for nx, ny, nz in block_divisions
    )
    np.savetxt(
        output_dir / "binder_parameter_vs_temperature.txt",
        np.column_stack((temperatures, binder_values)),
        header=header,
    )

    return critical_temperature



def plot_surface_tension(all_surface_tension, output_dir):
    """Plot mean surface tension versus target temperature."""
    if not all_surface_tension:
        return

    temperatures = np.array(sorted(all_surface_tension), dtype=float)
    gamma = np.array(
        [all_surface_tension[T][0] for T in temperatures],
        dtype=float,
    )
    gamma_error = np.array(
        [all_surface_tension[T][1] for T in temperatures],
        dtype=float,
    )

    # Watanabe Fig. 7 uses the reduced distance from the critical point.
    critical_temperature = 1.10
    epsilon = (critical_temperature - temperatures) / critical_temperature
    mask = (epsilon > 0.0) & (gamma > 0.0) & np.isfinite(gamma)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        epsilon[mask],
        gamma[mask],
        yerr=gamma_error[mask],
        fmt="o",
        capsize=3,
    )
    if np.count_nonzero(mask) >= 2:
        slope, intercept = np.polyfit(
            np.log(epsilon[mask]), np.log(gamma[mask]), 1
        )
        epsilon_fit = np.logspace(
            np.log10(np.min(epsilon[mask])),
            np.log10(np.max(epsilon[mask])), 100
        )
        ax.plot(
            epsilon_fit, np.exp(intercept) * epsilon_fit**slope,
            "--", label=rf"fit: $2\nu={slope:.3f}$"
        )
        ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"reduced temperature $\epsilon=(T_c-T)/T_c$")
    ax.set_ylabel(r"surface tension $\gamma$")
    ax.set_title(
        r"Surface tension from Watanabe Eq. (4), interface normal to $x$"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(
        output_dir / "surface_tension_vs_temperature.png",
        dpi=180,
    )
    plt.close(fig)

    np.savetxt(
        output_dir / "surface_tension_vs_temperature.txt",
        np.column_stack((temperatures, gamma, gamma_error)),
        header="temperature gamma gamma_standard_error",
    )

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
        default=[0.74, 0.80, 0.86, 0.90, 0.94, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.09, 1.10, 1.11, 1.12]
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

    # Compare all requested temperatures on common axes.
    plot_measurements(all_measurements, args.results)
    plot_local_density_histograms(
        all_density_histograms,
        args.density_min,
        args.density_max,
        args.results,
    )
    plot_local_density_vs_temperature(
        all_density_histograms,
        args.density_min,
        args.density_max,
        args.results,
    )
    plot_phase_boundary(
        all_density_histograms,
        args.density_min,
        args.density_max,
        args.results,
    )
    critical_temperature = plot_binder(all_binder, args.results)
    plot_surface_tension(all_surface_tension, args.results)

    if critical_temperature is None:
        print(
            "No Binder crossing found. Use several closely spaced "
            "temperatures around the critical region.",
            flush=True,
        )
    else:
        print(
            f"Estimated Binder-parameter crossing temperature: "
            f"{critical_temperature:.6f}",
            flush=True,
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
