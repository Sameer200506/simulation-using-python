"""
main_pack.py -- PyBatterySim Phase 2: 4-battery 2x2 pack with NTGK model.

Usage
-----
    python main_pack.py              # full 1511-second discharge, 3D view
    python main_pack.py --no3d       # skip interactive 3D window
    python main_pack.py --fast       # 300-second quick run for testing

Physics
-------
- NTGK electrochemical model (U_OCV polynomial fitted from LIB.xlsx)
- 4 x 26650 cells in 2x2 arrangement (4S1P)
- Natural convection on domain boundaries (Robin BC, h=5 W/m^2K)
- LOD/Yanenko solver, Numba-parallel Thomas algorithm
"""
import os
import sys
import datetime
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")

import config
from core.ntgk   import NTGKModel
from core.pack   import Pack
from core.solver import Solver
from visualization.viewer import (
    Viewer, plot_slice_xz, plot_history, plot_cell_comparison,
    plot_pack_top, animate_pack_top
)

# ── Unique Output Folder per Run ──────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE = config.OUTPUT_DIR
OUTPUT = os.path.join(OUTPUT_BASE, f"run_{timestamp}")
os.makedirs(OUTPUT, exist_ok=True)

XLSX    = "LIB.xlsx"

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--no3d",  action="store_true", help="Skip interactive 3D window")
ap.add_argument("--fast",  action="store_true", help="Run only 300 s for testing")
ap.add_argument("--xlsx",  action="store_true", help="Fit NTGK polynomial from LIB.xlsx instead of using pre-fitted constants")
args = ap.parse_args()


def main():
    print("=" * 64)
    print("  PyBatterySim -- Phase 2: NTGK 4-Cell Pack")
    print(f"  Saving results to: {OUTPUT}")
    print("=" * 64)

    # ── 1. Build NTGK model ───────────────────────────────────────────────────
    xlsx_file = XLSX if args.xlsx else None
    if xlsx_file:
        print("\n[1/5] Loading NTGK model and fitting from", xlsx_file, "...")
    else:
        print("\n[1/5] Initializing NTGK model (UDF mode - pre-fitted coefficients) ...")

    ntgk = NTGKModel(
        xlsx_path=xlsx_file,
        C_nom=4.5,          # Ah — typical 26650 NMC
        poly_order=5,
        n_cells_series=4,
    )
    ntgk.reset(dod=0.0)

    # Use config.T_END (1800s) as default simulation run time
    t_discharge = config.T_END if not args.fast else 300.0

    print(f"      Discharge duration: {t_discharge:.0f} s  "
          f"(I={ntgk._I_estimated:.2f} A, C_nom={ntgk.C_nom} Ah)")
    print(f"      R_int = {ntgk.R_int*1000:.1f} mOhm/cell")

    # ── 2. Build pack and grid ─────────────────────────────────────────────────
    print("\n[2/5] Building 2x2 pack geometry ...")
    pack = Pack(ntgk=ntgk, nx=50, ny=50, nz=50)
    grid = pack.build()

    g = grid
    print(f"      Grid: {g.nx}x{g.ny}x{g.nz} = {g.n_cells:,} cells")
    print(f"      Domain: {g.x_min*1e3:.0f} to {g.x_max*1e3:.0f} mm  x,y  |  "
          f"0 to {g.z_max*1e3:.0f} mm  z")
    batt_cells = int((g.material_id == 1).sum())
    print(f"      Battery cells: {batt_cells:,}  |  Air cells: {g.n_cells - batt_cells:,}")
    s0 = pack.stats()
    print(f"      Initial Q/cell: {s0['Q_per_cell_W']:.2f} W  "
          f"({s0['Q_per_cell_W']/((np.pi*config.RADIUS**2*config.HEIGHT)):.0f} W/m^3)")

    # ── 3. Run solver ──────────────────────────────────────────────────────────
    print(f"\n[3/5] Running NTGK-coupled thermal solver  "
          f"(dt={config.DT}s, t_end={t_discharge:.0f}s) ...")
    solver = Solver(grid, dt=config.DT, t_end=t_discharge, save_every=config.SAVE_EVERY)

    # Pack history for per-cell analysis
    pack_snapshots: list[dict] = []
    pack_history:   list[dict] = []

    def _snapshot_callback(slv, step):
        s = pack.stats()
        s["_t"] = slv.time
        pack_history.append(s)
        pack_snapshots.append({
            "t":   slv.time,
            "dod": ntgk.dod,
            "T":   grid.T.copy(),
        })

    solver.run(pack=pack, callback=_snapshot_callback, verbose=True)

    # ── 4. Outputs ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Saving outputs -> {OUTPUT}/")

    # XY top-down pack temperature map
    fig = plot_pack_top(grid, pack,
                        save_path=os.path.join(OUTPUT, "pack_top.png"))
    plt_close(fig)
    print(f"      -> {OUTPUT}/pack_top.png")

    # XZ cross-section
    fig = plot_slice_xz(grid,
                        save_path=os.path.join(OUTPUT, "pack_xz.png"))
    plt_close(fig)
    print(f"      -> {OUTPUT}/pack_xz.png")

    # Solver T history (aggregate)
    fig = plot_history(solver,
                       save_path=os.path.join(OUTPUT, "history.png"))
    plt_close(fig)
    print(f"      -> {OUTPUT}/history.png")

    # Per-cell comparison (NTGK analysis)
    if pack_history:
        fig = plot_cell_comparison(solver, pack_history,
                                   save_path=os.path.join(OUTPUT, "cell_comparison.png"))
        plt_close(fig)
        print(f"      -> {OUTPUT}/cell_comparison.png")

    # Animated top-down pack temperature
    if pack_snapshots:
        animate_pack_top(solver, pack_snapshots,
                         fps=8, save_path=os.path.join(OUTPUT, "pack_animation.gif"))
        print(f"      -> {OUTPUT}/pack_animation.gif")

    # Final summary
    final = pack.stats()
    hot   = pack.hottest_cell()
    print("\n" + "=" * 64)
    print("  PACK SIMULATION COMPLETE")
    print("=" * 64)
    print(f"  Final DOD         : {final['DOD']:.4f}")
    print(f"  Terminal voltage  : {final['V_terminal_V']:.3f} V/cell"
          f"  ({final['V_terminal_V']*4:.2f} V pack)")
    print(f"  Hottest cell      : Cell {hot[0]}  at  {hot[1]-273.15:.2f} C")
    print(f"  Cell temperatures : " +
          "  ".join(f"C{i}:{t:.1f}C" for i, t in enumerate(final['T_cells_C'])))
    print(f"  Heat per cell     : {final['Q_per_cell_W']:.2f} W")
    print(f"\n  Results in: {os.path.abspath(OUTPUT)}")

    # ── 5. Interactive 3D ─────────────────────────────────────────────────────
    if not args.no3d:
        print("\n[5/5] Launching 3D surface viewer ...")
        from visualization.viewer import plot_3d
        plot_3d(grid, pack=pack)
    else:
        print("\n[5/5] 3D render skipped (--no3d)")

    print("\nDone.")


def plt_close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
