"""Velocity-Verlet molecular dynamics for a Lennard-Jones fluid.

Running this file performs the two temperature runs requested in Exercise 7 and
writes trajectories, measurements, and plots to ``results``.
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


# Lennard-Jones reduced units: sigma = epsilon = mass = k_B = 1.
N = 125
L = 30.0
rho = N / L**3
dt = 0.005
rcut = 4.0
rcut2 = rcut**2
Ecut = 4.0 * (rcut**-12 - rcut**-6)


def init_positions(n_particles, box_length):
    """Put particles on a cubic lattice without overlapping them."""
    n_side = int(np.ceil(n_particles ** (1.0 / 3.0)))
    spacing = box_length / n_side
    coordinates = (np.arange(n_side) + 0.5) * spacing
    grid = np.array(np.meshgrid(coordinates, coordinates, coordinates,
                                indexing="ij")).reshape(3, -1).T
    return grid[:n_particles].copy()


def init_velocities(n_particles, temperature, rng):
    """Draw velocities, remove centre-of-mass motion, and set temperature."""
    velocities = rng.random((n_particles, 3)) - 0.5
    # Zero total momentum by removing the centre-of-mass velocity.
    velocities -= velocities.mean(axis=0)
    # Rescale so sum(v^2)/(3N) equals the requested reduced temperature.
    mean_squared_component = np.sum(velocities**2) / (3.0 * n_particles)
    velocities *= np.sqrt(temperature / mean_squared_component)
    return velocities


@njit
def compute_forces(positions, box_length):
    """Return truncated-and-shifted LJ forces and potential energy."""
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    rij = np.empty(3, dtype=np.float64)
    n_particles = positions.shape[0]
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            for k in range(3):
                # Minimum-image displacement r_i-r_j.
                rij[k] = positions[i, k] - positions[j, k]
                rij[k] -= box_length * np.rint(rij[k] / box_length)

            r2 = rij[0]**2 + rij[1]**2 + rij[2]**2
            # Interactions outside the cutoff are neglected.
            if r2 < rcut2:
                inv_r2 = 1.0 / r2
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                force_factor = (48.0 * inv_r12 - 24.0 * inv_r6) * inv_r2
                # Newton's third law: equal and opposite pair forces.
                for k in range(3):
                    pair_force = force_factor * rij[k]
                    forces[i, k] += pair_force
                    forces[j, k] -= pair_force
                potential_energy += 4.0 * (inv_r12 - inv_r6) - Ecut
    return forces, potential_energy


@njit
def calc_positionsv2v(positions, velocities, step_size, box_length):
    """Drift positions for one full step and wrap them into the box."""
    positions += velocities * step_size
    positions %= box_length
    return positions


@njit
def calc_velocitiesv2v(velocities, forces, step_size):
    """Apply one half-kick (unit particle mass)."""
    velocities += 0.5 * forces * step_size
    return velocities


@njit
def simulate(positions, velocities, box_length, step_size, n_steps,
             sample_every, trajectory_every):
    """Run velocity Verlet and retain sampled observables and trajectory."""
    forces, potential_energy = compute_forces(positions, box_length)
    n_samples = n_steps // sample_every + 1
    n_frames = n_steps // trajectory_every + 1
    measurements = np.empty((n_samples, 9))
    trajectory = np.empty((n_frames, positions.shape[0], 3))
    sample_index = 0
    frame_index = 0

    for step in range(n_steps + 1):
        if step % sample_every == 0:
            kinetic_energy = 0.5 * np.sum(velocities**2)
            temperature = 2.0 * kinetic_energy / (3.0 * positions.shape[0])
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
            trajectory[frame_index] = positions
            frame_index += 1
        if step == n_steps:
            break

        velocities = calc_velocitiesv2v(velocities, forces, step_size)
        positions = calc_positionsv2v(positions, velocities, step_size,
                                      box_length)
        forces, potential_energy = compute_forces(positions, box_length)
        velocities = calc_velocitiesv2v(velocities, forces, step_size)

    return positions, velocities, measurements, trajectory


def write_xyz(filename, trajectory, trajectory_every, step_size, element="Ar"):
    """Write a complete (overwriting) wrapped XYZ trajectory."""
    with open(filename, "w", encoding="utf-8") as handle:
        for frame_index, positions in enumerate(trajectory):
            handle.write(f"{positions.shape[0]}\n")
            time = frame_index * trajectory_every * step_size
            handle.write(f"time = {time:.6f}; box = {L:.6f}\n")
            for x, y, z in positions:
                handle.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")


def plot_measurements(all_measurements, output_dir):
    colors = {0.1: "tab:blue", 1.5: "tab:orange"}

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for temperature, values in all_measurements.items():
        label = rf"$T_0={temperature:g}$"
        color = colors.get(temperature)
        energy0 = values[0, 3]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--trajectory-every", type=int, default=500)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.1, 1.5])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    all_measurements = {}
    header = "step time temperature total_energy kinetic_energy potential_energy Px Py Pz"
    for run_index, temperature in enumerate(args.temperatures):
        print(f"Running T_init={temperature:g} for {args.steps} steps ...")
        rng = np.random.default_rng(args.seed + run_index)
        positions = np.ascontiguousarray(init_positions(N, L), dtype=np.float64)
        velocities = np.ascontiguousarray(
            init_velocities(N, temperature, rng), dtype=np.float64)
        positions, velocities, measurements, trajectory = simulate(
            positions, velocities, L, dt, args.steps, args.sample_every,
            args.trajectory_every)
        tag = f"T{temperature:g}".replace(".", "p")
        np.savetxt(args.results / f"measurements_{tag}.txt", measurements,
                   header=header)
        write_xyz(args.results / f"trajectory_{tag}.xyz", trajectory,
                  args.trajectory_every, dt)
        all_measurements[temperature] = measurements
        drift = np.max(np.abs(measurements[:, 3] - measurements[0, 3]))
        max_momentum = np.max(np.linalg.norm(measurements[:, 6:9], axis=1))
        print(f"  max |E-E0|={drift:.3e}; max |P|={max_momentum:.3e}")

    plot_measurements(all_measurements, args.results)
    print(f"Results written to {args.results.resolve()}")


if __name__ == "__main__":
    main()
