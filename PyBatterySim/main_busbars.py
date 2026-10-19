"""
main_busbars.py -- PyBatterySim Phase 3: Pack with Busbars and Terminals.

Geometry:
  4 x 26650 cells in 2x2 formation (4S1P)
  + Nickel terminal disks (1 mm) at top/bottom of each cell
  + 3 Copper busbars connecting cells in series

Heat generation (pure physics, no Excel):
  Battery cell  : NTGK model  (pre-fitted U_OCV polynomial, I^2*R_int + entropic)
  Terminal cap  : I^2 * R_contact  (R_contact = 2 mOhm per weld)
  Busbar        : I^2 * R_busbar   (R_busbar = rho_Cu * L / A)

Usage:
  python main_busbars.py            # full 1800s simulation + 3D view
  python main_busbars.py --no3d     # skip 3D window
  python main_busbars.py --fast     # 300s quick run
"""
import os
import sys
import datetime
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
from core.ntgk          import NTGKModel
from core.pack_busbars  import PackWithBusbars
from core.solver        import Solver
from visualization.viewer import THERMAL_CMAP

OUTPUT_BASE = config.OUTPUT_DIR
timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT      = os.path.join(OUTPUT_BASE, f"busbars_{timestamp}")
os.makedirs(OUTPUT, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--no3d", action="store_true")
ap.add_argument("--fast", action="store_true")
args = ap.parse_args()


def _C(K):
    return np.asarray(K) - 273.15


def main():
    print("=" * 64)
    print("  PyBatterySim -- Phase 3: Pack + Busbars + Terminals")
    print(f"  Saving results to: {OUTPUT}")
    print("=" * 64)

    # â”€â”€ 1. NTGK model (no Excel required) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[1/5] Initializing NTGK model (UDF mode, pre-fitted coefficients) ...")
    ntgk = NTGKModel()        # no xlsx_path â†’ uses hardcoded polynomial
    ntgk.reset(dod=0.0)
    I = ntgk._I_estimated

    t_end = 300.0 if args.fast else config.T_END
    print(f"      I = {I:.2f} A   C_nom = {ntgk.C_nom} Ah   "
          f"R_int = {ntgk.R_int*1000:.1f} mOhm/cell   t_end = {t_end:.0f} s")

    # â”€â”€ 2. Build pack geometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[2/5] Building pack geometry (cells + terminals + busbars) ...")
    pack = PackWithBusbars(ntgk, nx=50, ny=50, nz=60)
    grid = pack.build()

    g = grid
    batt_cells = int((g.material_id == 1).sum())
    term_cells = int((g.material_id == 2).sum())
    bus_cells  = int((g.material_id == 3).sum())

    print(f"      Grid : {g.nx}Ã—{g.ny}Ã—{g.nz} = {g.n_cells:,} cells")
    print(f"      Domain x,y : {g.x_min*1e3:.0f} â†’ {g.x_max*1e3:.0f} mm  "
          f"z : {g.z_min*1e3:.1f} â†’ {g.z_max*1e3:.1f} mm")
    print(f"      Battery cells  : {batt_cells:,}")
    print(f"      Terminal cells : {term_cells:,}")
    print(f"      Busbar cells   : {bus_cells:,}")
    s0 = pack.stats()
    print(f"\n      Heat budget at t=0:")
    print(f"        Q per cell    = {s0['Q_cell_W']:.3f} W")
    print(f"        Q contact/weld= {s0['Q_contact_W']:.3f} W  (per terminal, 2 mÎ©)")
    print(f"        Q busbars tot = {s0['Q_busbar_total_W']*1e3:.2f} mW  "
          f"(copper, negligible)")
    print(f"        Q total pack  = {s0['Q_total_W']:.2f} W")

    # â”€â”€ 3. Solver run â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[3/5] Running coupled thermal solver ...")
    solver = Solver(grid, dt=config.DT, t_end=t_end,
                    t_amb=config.T_AMB, h_conv=config.H_CONV,
                    save_every=config.SAVE_EVERY)

    history: list[dict] = []
    snapshots: list[dict] = []

    def _cb(slv, step):
        s = pack.stats()
        s["_t"] = slv.time
        history.append(s)
        if step % (config.SAVE_EVERY * 2) == 0:
            snapshots.append({"t": slv.time, "T": grid.T.copy()})

    solver.run(pack=pack, callback=_cb, verbose=True)

    # â”€â”€ 4. Plots â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[4/5] Saving plots to {OUTPUT}/ ...")

    _plot_heat_budget(history, os.path.join(OUTPUT, "heat_budget.png"))
    _plot_temperatures(history, os.path.join(OUTPUT, "temperatures.png"))
    _plot_pack_top(grid, pack, os.path.join(OUTPUT, "pack_top.png"))
    _plot_xz_slice(grid, pack, os.path.join(OUTPUT, "pack_xz.png"))
    _plot_heat_generation_xz(grid, pack, os.path.join(OUTPUT, "heat_generation_xz.png"))
    _plot_busbar_close_up(grid, pack, os.path.join(OUTPUT, "busbar_closeup.png"))

    for f in ["heat_budget.png", "temperatures.png", "pack_top.png",
              "pack_xz.png", "heat_generation_xz.png", "busbar_closeup.png"]:
        print(f"      -> {OUTPUT}/{f}")

    # â”€â”€ 5. Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    final = pack.stats()
    print("\n" + "=" * 64)
    print("  SIMULATION COMPLETE")
    print("=" * 64)
    print(f"  DOD                   : {final['DOD']:.4f}")
    print(f"  Pack voltage          : {final['V_terminal_V']*4:.2f} V")
    print(f"  Heat per cell (NTGK)  : {final['Q_cell_W']:.3f} W")
    print(f"  Heat per terminal weld: {final['Q_contact_W']:.3f} W  "
          f"({final['Q_contact_W']/final['Q_cell_W']*100:.1f}% of cell heat)")
    print(f"  Heat all busbars      : {final['Q_busbar_total_W']*1e3:.2f} mW  "
          f"(copper, nearly zero)")
    print(f"  Hottest component     : {final['hottest_component']}  "
          f"@ {final['hottest_T_C']:.2f} Â°C")
    print(f"  Cell temps (max Â°C)   : "
          + "  ".join(f"C{i}:{t:.1f}" for i, t in enumerate(final['T_cells_C'])))

    if not args.no3d:
        print("\n[5/5] Launching 3D viewer ...")
        _show_3d(grid)
    else:
        print("\n[5/5] 3D render skipped (--no3d)")


# â”€â”€ Plot helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _plot_heat_budget(history, path):
    """Stacked area chart: heat breakdown per component over time."""
    t   = np.array([h["_t"] for h in history])
    Q_c = np.array([h["Q_cell_W"]         * 4  for h in history])   # 4 cells
    Q_t = np.array([h["Q_contact_W"]      * 8  for h in history])   # 8 welds
    Q_b = np.array([h["Q_busbar_total_W"] * 1e3 for h in history])  # mW â†’ still small

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top: stacked area
    ax1.stackplot(t, Q_c, Q_t,
                  labels=["Battery cells (Ã—4, NTGK)",
                          "Terminal welds (Ã—8, contact R)"],
                  colors=["#EF5350", "#FFA726"], alpha=0.85)
    ax1.set_ylabel("Heat Generation  (W)", fontsize=11)
    ax1.set_title("Heat Budget vs Time â€” Pack with Busbars & Terminals\n"
                  "(Pure IÂ²R physics, no Excel)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Bottom: busbar heat (mW scale)
    ax2.plot(t, Q_b, color="#42A5F5", lw=2, label="Busbars total (mW, copper)")
    ax2.set_ylabel("Busbar Heat  (mW)", fontsize=11)
    ax2.set_xlabel("Time  (s)", fontsize=11)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_temperatures(history, path):
    """4-panel: DOD, voltage, cell T_max, hottest component."""
    t     = np.array([h["_t"] for h in history])
    dod   = np.array([h["DOD"] for h in history])
    volt  = np.array([h["V_terminal_V"] * 4 for h in history])
    T_max = np.array([h["T_max_C"] for h in history])
    T_hot = np.array([h["hottest_T_C"] for h in history])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(t, dod, lw=2, color="#7E57C2")
    axes[0, 0].set_title("Depth of Discharge"); axes[0, 0].set_ylabel("DOD")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, volt, lw=2, color="#1565C0")
    axes[0, 1].set_title("Pack Terminal Voltage"); axes[0, 1].set_ylabel("V (V)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, T_max, lw=2, color="#EF5350", label="Cell T_max")
    axes[1, 0].axhline(config.T_AMB - 273.15, color="grey", lw=1, linestyle=":", label="T_amb")
    axes[1, 0].set_title("Max Battery Cell Temperature")
    axes[1, 0].set_ylabel("°C"); axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(t, T_hot, lw=2, color="#FF7043",
                    label="Hottest component (cell/terminal/busbar)")
    axes[1, 1].set_title("Hottest Voxel in Entire Pack")
    axes[1, 1].set_ylabel("°C"); axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Time  (s)")

    fig.suptitle("PyBatterySim — Temperature & Electrical State\n"
                 "4S1P Pack + Busbars + Terminals", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pack_top(grid, pack, path):
    """Top-down heat map at mid-height showing cells, terminals, busbars."""
    from core.pack_busbars import CELL_CENTRES, _R, MAT_BUSBAR, MAT_TERMINAL
    z_idx = grid.nz // 2
    T_sl  = _C(grid.T[:, :, z_idx])
    mat_sl = grid.material_id[:, :, z_idx]

    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(T_sl.T, origin="lower",
                   extent=[grid.x_min*1e3, grid.x_max*1e3,
                           grid.y_min*1e3, grid.y_max*1e3],
                   cmap=THERMAL_CMAP, aspect="equal")
    plt.colorbar(im, ax=ax, label="Temperature (°C)", fraction=0.046)

    for i, (cx, cy) in enumerate(CELL_CENTRES):
        ax.add_patch(plt.Circle((cx*1e3, cy*1e3), _R*1e3,
                                fill=False, edgecolor="white", lw=1.5, linestyle="--"))
        ax.text(cx*1e3, cy*1e3, f"Cell {i}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")

    ax.set_title(f"Pack Temperature Map (mid-height z={grid.z[z_idx]*1e3:.0f} mm)\n"
                 f"Hottest: {pack.hottest_component()[0]} @ "
                 f"{pack.hottest_component()[1]:.1f}°C   DOD={pack.ntgk.dod:.3f}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_xz_slice(grid, pack, path):
    """XZ cross-section through the cell centres (y = _CX)."""
    from core.pack_busbars import _CX
    y_idx = int(np.argmin(np.abs(grid.y - _CX)))
    T_sl  = _C(grid.T[:, y_idx, :])

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(T_sl.T, origin="lower",
                   extent=[grid.x_min*1e3, grid.x_max*1e3,
                           grid.z_min*1e3, grid.z_max*1e3],
                   cmap=THERMAL_CMAP, aspect="auto")
    plt.colorbar(im, ax=ax, label="Temperature (deg C)", fraction=0.035)
    ax.set_title(f"XZ Cross-Section (Temperature)  y = {grid.y[y_idx]*1e3:.1f} mm\n"
                 "Cells + Terminal caps + Busbars")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_heat_generation_xz(grid, pack, path):
    """XZ cross-section of volumetric heat source (y = _CX)."""
    from core.pack_busbars import _CX
    from matplotlib.colors import LogNorm
    y_idx = int(np.argmin(np.abs(grid.y - _CX)))
    Q_sl  = grid.Q[:, y_idx, :]

    fig, ax = plt.subplots(figsize=(8, 10))
    # Handle zeros safely for log norm
    Q_safe = np.clip(Q_sl, 1e1, None)
    im = ax.imshow(Q_safe.T, origin="lower",
                   extent=[grid.x_min*1e3, grid.x_max*1e3,
                           grid.z_min*1e3, grid.z_max*1e3],
                   cmap="YlOrRd", norm=LogNorm(vmin=1e2, vmax=Q_safe.max()), aspect="auto")
    plt.colorbar(im, ax=ax, label="Volumetric Heat Source (W/m^3)")
    ax.set_title(f"XZ Heat Source Distribution (Q)  y = {grid.y[y_idx]*1e3:.1f} mm\n"
                 "Battery cells + Nickel caps + Copper busbars")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_busbar_close_up(grid, pack, path):
    """Zoomed XZ slice showing terminal caps and busbar region."""
    from core.pack_busbars import _H, _CX
    y_idx = int(np.argmin(np.abs(grid.y - _CX)))

    # Focus on top busbar region: z in [H-5mm, H+6mm]
    z_lo = (_H - 0.005)
    z_hi = (_H + 0.006)
    iz0  = int(np.searchsorted(grid.z, z_lo))
    iz1  = int(np.searchsorted(grid.z, z_hi))
    iz0, iz1 = max(0, iz0), min(grid.nz, iz1)

    T_region   = _C(grid.T[:, y_idx, iz0:iz1])
    mat_region = grid.material_id[:, y_idx, iz0:iz1]

    ext = [grid.x_min*1e3, grid.x_max*1e3,
           grid.z[iz0]*1e3, grid.z[min(iz1, grid.nz-1)]*1e3]


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    im1 = ax1.imshow(T_region.T, origin="lower", extent=ext,
                     cmap=THERMAL_CMAP, aspect="auto")
    plt.colorbar(im1, ax=ax1, label="Temperature (°C)")
    ax1.set_title("Temperature -- Busbar Region (top)")
    ax1.set_xlabel("x (mm)"); ax1.set_ylabel("z (mm)")

    im2 = ax2.imshow(mat_region.T, origin="lower", extent=ext,
                     cmap=plt.cm.Set1, aspect="auto", vmin=0, vmax=3)
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
    cbar2.set_ticklabels(["Air", "Battery", "Terminal", "Busbar"])
    ax2.set_title("Material Map -- Busbar Region (top)")
    ax2.set_xlabel("x (mm)"); ax2.set_ylabel("z (mm)")

    fig.suptitle("Close-Up: Terminal Caps & Copper Busbar", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _show_3d(grid):
    try:
        import pyvista as pv
    except ImportError:
        print("[Viewer] pyvista not installed - skipping 3D view.")
        return

    mesh = pv.ImageData()
    mesh.dimensions = (grid.nx+1, grid.ny+1, grid.nz+1)
    mesh.origin     = (grid.x_min, grid.y_min, grid.z_min)
    mesh.spacing    = (grid.dx,    grid.dy,    grid.dz)
    mesh.cell_data["Temperature_C"] = _C(grid.T).ravel(order="F")
    mesh.cell_data["Heat_Gen_W_m3"] = grid.Q.ravel(order="F")
    mesh.cell_data["material"]      = grid.material_id.ravel(order="F").astype(float)

    # 1x2 viewport: Left = Temp, Right = Heat Gen
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 850))

    # Viewport 0: Temperature (Only render Battery to keep it clear)
    pl.subplot(0, 0)
    for mat_id, name, color, opacity in [
        (1, "Battery",  "coolwarm", 1.0),
    ]:
        clipped = mesh.threshold([mat_id - 0.4, mat_id + 0.4], scalars="material", preference="cell")
        if clipped.n_cells > 0:
            surf = clipped.extract_surface(algorithm='dataset_surface')
            pl.add_mesh(surf, scalars="Temperature_C", cmap="coolwarm",
                        smooth_shading=True, opacity=opacity, label=name,
                        scalar_bar_args={"title": "Temp (deg C)", "fmt": "%.1f"})
    pl.add_legend()
    pl.add_text("Temperature Distribution", position="upper_edge", font_size=12)
    pl.set_background("black")
    pl.show_axes()

    # Viewport 1: Volumetric Heat source (Q) (Only render Battery to keep it clear)
    pl.subplot(0, 1)
    for mat_id, name, color, opacity in [
        (1, "Battery",  "YlOrRd", 1.0),
    ]:
        clipped = mesh.threshold([mat_id - 0.4, mat_id + 0.4], scalars="material", preference="cell")
        if clipped.n_cells > 0:
            surf = clipped.extract_surface(algorithm='dataset_surface')
            pl.add_mesh(surf, scalars="Heat_Gen_W_m3", cmap="YlOrRd",
                        smooth_shading=True, opacity=opacity, label=name,
                        scalar_bar_args={"title": "Heat Gen (W/m^3)", "fmt": "%.1e"})
    pl.add_text("Volumetric Heat Source (Q)", position="upper_edge", font_size=12)
    pl.set_background("black")
    pl.show_axes()

    pl.link_views()  # synchronize rotations/pans between left and right viewports
    pl.show(title="PyBatterySim -- Temperature (left) & Heat Source (right)")


if __name__ == "__main__":
    main()
