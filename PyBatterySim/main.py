"""
main.py — PyBatterySim Phase 1 entry point.

Run:
    python main.py

What it does
------------
1. Builds a 26650 battery on a Cartesian voxel grid.
2. Runs the explicit finite-volume heat solver.
3. Saves temperature slice images, a history plot, and an animation.
4. Optionally opens an interactive 3-D render (requires pyvista).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless by default; change to "TkAgg" for GUI windows

import config
from core.battery import Battery
from core.grid import Grid
from core.solver import Solver
from visualization.viewer import Viewer

OUTPUT = config.OUTPUT_DIR
os.makedirs(OUTPUT, exist_ok=True)


def main():
    print("=" * 60)
    print("  PyBatterySim -- Phase 1: Thermal Foundation")
    print("=" * 60)

    # ── 1. Build grid + stamp battery geometry ────────────────────────────────
    print("\n[1/4] Building grid and battery geometry ...")
    grid    = Grid()
    battery = Battery(grid=grid)
    battery.build()

    print(f"      {grid}")
    info = battery.info()
    print(f"      Battery  r={info['radius_mm']:.1f} mm  "
          f"h={info['height_mm']:.1f} mm  "
          f"Q={info['heat_gen_W_m3']:.0f} W/m3  "
          f"(total {info['total_heat_W']:.2f} W)")
    stats = grid.stats()
    print(f"      Cells: {stats['battery_cells']:,} battery / "
          f"{stats['air_cells']:,} air  "
          f"(T_init = {stats['T_min_K'] - 273.15:.1f} C)")

    # ── 2. Run solver ─────────────────────────────────────────────────────────
    print(f"\n[2/4] Running solver  (dt={config.DT} s, t_end={config.T_END} s) ...")
    solver = Solver(grid)
    solver.run(verbose=True)

    # ── 3. Save outputs ───────────────────────────────────────────────────────
    print("\n[3/4] Saving outputs ...")
    viewer = Viewer(solver)

    viewer.show_slice_xy(save=os.path.join(OUTPUT, "slice_xy.png"))
    print(f"      -> {OUTPUT}/slice_xy.png")

    viewer.show_slice_xz(save=os.path.join(OUTPUT, "slice_xz.png"))
    print(f"      -> {OUTPUT}/slice_xz.png")

    viewer.show_history(save=os.path.join(OUTPUT, "history.png"))
    print(f"      -> {OUTPUT}/history.png")

    viewer.animate(save=os.path.join(OUTPUT, "animation.gif"), fps=8)
    print(f"      -> {OUTPUT}/animation.gif")

    # ── 4. Optional 3-D interactive render ───────────────────────────────────
    print("\n[4/4] 3-D render ...")
    if "--no3d" in sys.argv:
        print("      Skipped (--no3d flag).")
    else:
        try:
            import pyvista  # noqa: F401
            matplotlib.use("TkAgg")   # switch to interactive backend for 3D window
            viewer.show_3d()
        except ImportError:
            print("      pyvista not installed — skipping.  "
                  "Run:  pip install pyvista vtk")

    print("\nDone.  Results in:", os.path.abspath(OUTPUT))


if __name__ == "__main__":
    main()
