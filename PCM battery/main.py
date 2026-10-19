"""
main.py
-------
Entry point for the 4-cell 26650 Li-ion / PCM pack thermal simulation.

Workflow
────────
  1. Parse command-line arguments.
  2. Generate the 2-D FEM mesh (gmsh → DOLFINx).
  3. Run the transient solver WITH PCM (RT-42 paraffin).
  4. Run the transient solver WITHOUT PCM (air surroundings – reference).
  5. Post-process: 5 figures + results.csv + printed summary.

Usage
─────
  python main.py                        # default parameters
  python main.py --dt 15 --t_end 1800  # 30-min run, 15 s steps
  python main.py --output my_results    # custom output directory
  python main.py --help                 # show all options

Output tree (inside --output directory)
────────────────────────────────────────
  mesh/
    mesh.xdmf               ← geometry + subdomain tags for ParaView
    mesh.h5
  pcm/
    temperature.xdmf        ← transient T snapshots (with PCM)
    temperature.h5
  nopcm/
    temperature_nopcm.xdmf  ← transient T snapshots (no PCM / air)
    temperature_nopcm.h5
  results/
    fig1_battery_temperature.png
    fig2_pcm_temperature.png
    fig3_max_battery_temp.png
    fig4_heat_absorbed.png
    fig5_liquid_fraction.png
    results.csv
"""

import argparse
import time
from pathlib import Path

from mesh       import generate_mesh
from solver     import BatteryPCMSolver
from postprocess import PostProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transient PCM battery-pack thermal simulation (FEniCSx)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dt", type=float, default=30.0,
        help="Time step [s]",
    )
    p.add_argument(
        "--t_end", type=float, default=1800.0,
        help="Total simulation time [s]",
    )
    p.add_argument(
        "--output", type=str, default="output",
        help="Root output directory",
    )
    p.add_argument(
        "--mesh_bat", type=float, default=0.0015,
        help="Target element size at battery surfaces [m]",
    )
    p.add_argument(
        "--mesh_pcm", type=float, default=0.004,
        help="Target element size in PCM bulk [m]",
    )
    p.add_argument(
        "--save_every", type=int, default=5,
        help="Write XDMF snapshot every N timesteps",
    )
    return p.parse_args()


def banner(text: str) -> None:
    """Print a section header to stdout."""
    sep = "─" * 60
    print(f"\n{sep}\n  {text}\n{sep}")


def main() -> None:
    args       = parse_args()
    output_dir = Path(args.output)

    print("╔" + "═" * 58 + "╗")
    print("║  PCM Battery Pack – Transient Thermal Simulation       ║")
    print("║  FEniCSx (DOLFINx)  ·  gmsh  ·  NumPy  ·  Matplotlib  ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Δt      = {args.dt:>8.1f} s                              ║")
    print(f"║  t_end   = {args.t_end:>8.1f} s  ({args.t_end/3600:.2f} h)                 ║")
    print(f"║  output  = {str(output_dir):<47s}  ║")
    print("╚" + "═" * 58 + "╝")

    t0_total = time.perf_counter()

    # ── Step 1: Mesh generation ───────────────────────────────────────────────
    banner("STEP 1 / 4 – Mesh generation")
    t0 = time.perf_counter()
    msh, cell_tags, facet_tags = generate_mesh(
        output_dir    = output_dir / "mesh",
        mesh_size_bat = args.mesh_bat,
        mesh_size_pcm = args.mesh_pcm,
    )
    print(f"  Elapsed: {time.perf_counter() - t0:.1f} s")

    # ── Step 2: Simulation WITH PCM ───────────────────────────────────────────
    banner("STEP 2 / 4 – Transient solver  (with PCM)")
    t0 = time.perf_counter()
    solver_pcm = BatteryPCMSolver(
        msh, cell_tags, facet_tags,
        dt         = args.dt,
        t_end      = args.t_end,
        output_dir = output_dir / "pcm",
        use_pcm    = True,
        save_every = args.save_every,
    )
    ts_pcm = solver_pcm.run()
    print(f"  Elapsed: {time.perf_counter() - t0:.1f} s")

    # ── Step 3: Simulation WITHOUT PCM (air reference) ───────────────────────
    banner("STEP 3 / 4 – Transient solver  (no PCM – air reference)")
    t0 = time.perf_counter()
    solver_nopcm = BatteryPCMSolver(
        msh, cell_tags, facet_tags,
        dt         = args.dt,
        t_end      = args.t_end,
        output_dir = output_dir / "nopcm",
        use_pcm    = False,
        save_every = args.save_every,
    )
    ts_nopcm = solver_nopcm.run()
    print(f"  Elapsed: {time.perf_counter() - t0:.1f} s")

    # ── Step 4: Post-processing ───────────────────────────────────────────────
    banner("STEP 4 / 4 – Post-processing")
    pp = PostProcessor(
        ts_pcm     = ts_pcm,
        ts_nopcm   = ts_nopcm,
        output_dir = output_dir / "results",
    )
    pp.plot_all()
    pp.export_csv()
    pp.print_summary()

    total = time.perf_counter() - t0_total
    print(f"Total wall time: {total:.1f} s\n")
    print("ParaView files:")
    print(f"  {output_dir / 'mesh'  / 'mesh.xdmf'}")
    print(f"  {output_dir / 'pcm'   / 'temperature.xdmf'}")
    print(f"  {output_dir / 'nopcm' / 'temperature_nopcm.xdmf'}")
    print(f"\nResults: {output_dir / 'results' / 'results.csv'}")


if __name__ == "__main__":
    main()
