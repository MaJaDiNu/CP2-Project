"""Calculate and plot radial distribution functions from XYZ trajectories."""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numba import njit


def read_xyz(filename):
    """Read all frames from an XYZ trajectory into a 3-D NumPy array."""
    frames = []
    n_particles = None
    with open(filename, "r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            frame_n = int(line.strip())
            if n_particles is None:
                n_particles = frame_n
            elif frame_n != n_particles:
                raise ValueError("Particle count changes between XYZ frames")
            if not handle.readline():
                raise ValueError("Missing XYZ comment line")
            positions = np.empty((frame_n, 3), dtype=np.float64)
            for i in range(frame_n):
                parts = handle.readline().split()
                if len(parts) < 4:
                    raise ValueError("Incomplete XYZ particle record")
                positions[i] = (float(parts[1]), float(parts[2]), float(parts[3]))
            frames.append(positions)
    if not frames:
        raise ValueError(f"No frames found in {filename}")
    return n_particles, np.ascontiguousarray(np.asarray(frames))


@njit
def compute_g(positions, box_length, n_particles, dr, dmax, nbins):
    """Histogram pair distances using periodic minimum images and normalise."""
    counts = np.zeros(nbins)
    n_frames = positions.shape[0]
    dx = np.empty(3, dtype=np.float64)
    # Accumulate distances independently over every trajectory frame.
    for frame in range(n_frames):
        for i in range(n_particles):
            # Visit each unordered pair once, then count both neighbours.
            for j in range(i + 1, n_particles):
                for component in range(3):
                    dx[component] = (positions[frame, i, component]
                                     - positions[frame, j, component])
                    dx[component] -= box_length * np.rint(
                        dx[component] / box_length)
                distance = np.sqrt(dx[0]**2 + dx[1]**2 + dx[2]**2)
                if distance < dmax:
                    counts[int(distance / dr)] += 2.0

    r_values = (np.arange(nbins) + 0.5) * dr
    shell_volumes = (4.0 / 3.0) * np.pi * (
        (r_values + dr / 2.0)**3 - (r_values - dr / 2.0)**3)
    number_density = n_particles / box_length**3
    # Ideal-gas expectation: frames * particles * density * shell volume.
    g_normalized = counts / (n_frames * n_particles * number_density
                             * shell_volumes)
    return r_values, g_normalized


def main():
    script_dir = Path(__file__).parent
    default_results = script_dir / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectories", type=Path, nargs="*",
                        default=[default_results / "trajectory_T0p1.xyz",
                                 default_results / "trajectory_T1p5.xyz"])
    parser.add_argument("--box-length", type=float, default=30.0)
    parser.add_argument("--dr", type=float, default=0.1)
    parser.add_argument("--results", type=Path, default=default_results)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    dmax = args.box_length / 2.0
    nbins = int(dmax / args.dr)
    fig, ax = plt.subplots(figsize=(8, 5))
    for trajectory_file in args.trajectories:
        n_particles, positions = read_xyz(trajectory_file)
        r_values, g_values = compute_g(positions, args.box_length,
                                       n_particles, args.dr, dmax, nbins)
        output_file = args.results / f"g_r_{trajectory_file.stem}.txt"
        np.savetxt(output_file, np.column_stack((r_values, g_values)),
                   header="r g(r)", fmt="%.6f %.6f")
        label = trajectory_file.stem.replace("trajectory_", "").replace("p", ".")
        ax.plot(r_values, g_values, linewidth=1.4, label=label)
        print(f"Analysed {positions.shape[0]} frames from {trajectory_file}")

    ax.axhline(1.0, color="0.45", linestyle="--", linewidth=1,
               label="ideal gas")
    ax.set(xlabel="r", ylabel="g(r)", title="Radial distribution functions",
           xlim=(0.0, dmax))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.results / "radial_distribution.png", dpi=180)
    plt.close(fig)
    print(f"Results written to {args.results.resolve()}")


if __name__ == "__main__":
    main()
