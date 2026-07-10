"""Velocity-Verlet molecular dynamics for a Lennard-Jones fluid.

Running this file performs the two temperature runs requested in Exercise 7 and
writes trajectories, measurements, and plots to ``results``.
"""

from pathlib import Path
import argparse

import matplotlib
# Use a non-interactive backend so figures can also be created without a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


# Lennard-Jones reduced units: sigma = epsilon = mass = k_B = 1.
# Consequently, all lengths, energies, times, and temperatures below are
# dimensionless reduced quantities.
N = 125

Lx = 12.0
Ly = 12.0
Lz = 12.0

box = np.array([Lx, Ly, Lz], dtype=np.float64)

rcut = 2.5
dt = 0.01
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

def init_positions(n_particles, box):
    """Put particles on a cubic lattice without overlapping them."""
    # Use enough lattice sites to hold every particle.
    n_side = int(np.ceil(n_particles ** (1.0 / 3.0)))
    # Offset by half a lattice spacing so no particle lies on a box boundary.
    x_coordinates=(np.arange(n_side)+0.5)*box[0]/n_side
    y_coordinates=(np.arange(n_side)+0.5)*box[1]/n_side
    z_coordinates=(np.arange(n_side)+0.5)*box[2]/n_side
    
    
    # meshgrid forms every possible combination of the x, y, and z coordinates.
    grid = np.array(np.meshgrid(x_coordinates, y_coordinates, z_coordinates,
                                indexing="ij")).reshape(3, -1).T
    return grid[:n_particles].copy()


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
def compute_forces(positions, box):
    """Return truncated-and-shifted LJ forces and potential energy."""
    # A separate force vector is accumulated for each particle.
    forces = np.zeros_like(positions)
    potential_energy = 0.0
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
                if r2 < rcut2:
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
    return forces, potential_energy


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
             sample_every, trajectory_every, T_target, tau_t):
    """Run velocity Verlet and retain sampled observables and trajectory."""
    # Forces at the initial positions are needed for the first half-kick.
    forces, potential_energy = compute_forces(positions,box)
    # Preallocate output arrays because allocation inside a JIT loop is costly.
    n_samples = n_steps // sample_every + 1
    n_frames = n_steps // trajectory_every + 1
    measurements = np.empty((n_samples, 9))
    trajectory = np.empty((n_frames, positions.shape[0], 3))
    sample_index = 0
    frame_index = 0

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
        if step == n_steps:
            break

        # Velocity Verlet sequence: half-kick, full drift, force update,
        # and a second half-kick using the force at the new positions.
        velocities = calc_velocitiesv2v(velocities, forces, step_size)
        positions = calc_positionsv2v(positions, velocities, step_size,
                                      box)
        forces, potential_energy = compute_forces(positions, box)
        velocities = calc_velocitiesv2v(velocities, forces, step_size)

        # Couple the system to the canonical heat bath after a complete
        # velocity-Verlet step, when velocities are defined at an integer time.
        velocities = bussi_thermostat(velocities, T_target, step_size, tau_t)

    return positions, velocities, measurements, trajectory


def write_xyz(filename, trajectory, trajectory_every, step_size, element="Ar"):
    """Write a complete (overwriting) wrapped XYZ trajectory."""
    # Opening with mode "w" prevents a new run being appended to an old one.
    with open(filename, "w", encoding="utf-8") as handle:
        for frame_index, positions in enumerate(trajectory):
            # XYZ frames begin with particle count followed by one comment line.
            handle.write(f"{positions.shape[0]}\n")
            time = frame_index * trajectory_every * step_size
            handle.write(f"time = {time:.6f}; box = {Lx:.6f}\n")
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


def main():
    # Command-line options allow shorter tests without editing the source code.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--trajectory-every", type=int, default=500)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.8])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--thermostat-tau", type=float, default=1.0,
                        help="Bussi thermostat relaxation time; <= 0 disables it")
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()

    if args.sample_every <= 0:
        raise ValueError("--sample-every must be a positive integer")

    if args.trajectory_every <= 0:
        raise ValueError("--trajectory-every must be a positive integer")

    args.results.mkdir(parents=True, exist_ok=True)

    all_measurements = {}
    header = "step time temperature total_energy kinetic_energy potential_energy Px Py Pz"
    for run_index, temperature in enumerate(args.temperatures):
        print(f"Running T_target={temperature:g} for {args.steps} steps ...")
        # A reproducible but different random velocity set is used for each run.
        rng = np.random.default_rng(args.seed + run_index)
        # C-contiguous float arrays give Numba efficient memory access.
        positions = np.ascontiguousarray(init_positions(N, box), dtype=np.float64)
        velocities = np.ascontiguousarray(
            init_velocities(N, temperature, rng), dtype=np.float64)
        positions, velocities, measurements, trajectory = simulate(
            positions, velocities, box, dt, args.steps, args.sample_every,
            args.trajectory_every, temperature, args.thermostat_tau)
        tag = f"T{temperature:g}".replace(".", "p")
        # Save numerical data separately from the figures for later analysis.
        np.savetxt(args.results / f"measurements_{tag}.txt", measurements,
                   header=header)
        write_xyz(args.results / f"trajectory_{tag}.xyz", trajectory,
                  args.trajectory_every, dt)
        all_measurements[temperature] = measurements
        drift = np.max(np.abs(measurements[:, 3] - measurements[0, 3]))
        max_momentum = np.max(np.linalg.norm(measurements[:, 6:9], axis=1))
        print(f"  max |E-E0|={drift:.3e}; max |P|={max_momentum:.3e}")

    # Compare all requested temperatures on common axes.
    plot_measurements(all_measurements, args.results)
    print(f"Results written to {args.results.resolve()}")


if __name__ == "__main__":
    main()
