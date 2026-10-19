"""
visualization/viewer.py -- All post-processing and visualisation for PyBatterySim.

Changes in Phase 2
------------------
- 3D rendering: switched from add_volume() (slow) to extract_surface() + add_mesh()
  (no GPU transparency overhead -- renders 10-100x faster)
- New pack-level plots: top-down heat map, per-cell temperature comparison,
  hotspot annotation
- All existing single-cell plots still work unchanged
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import config

# ── Custom thermal colormap (COMSOL-style) ────────────────────────────────────
_THERMAL_COLORS = [
    "#05006B", "#0040FF", "#00BFFF", "#00FF80",
    "#FFFF00", "#FF8000", "#FF0000", "#800000",
]
THERMAL_CMAP = LinearSegmentedColormap.from_list("thermal", _THERMAL_COLORS, N=256)


def _C(K):
    return K - 273.15


# ─────────────────────────────────────────────────────────────────────────────
# Single-cell 2-D slice plots (unchanged from Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def plot_slice_xy(grid, z_index=None, title="Temperature -- XY Slice",
                  save_path=None):
    if z_index is None:
        z_index = grid.nz // 2
    T_slice = _C(grid.T[:, :, z_index])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(T_slice.T, origin="lower",
                   extent=[grid.x_min*1e3, grid.x_max*1e3,
                           grid.y_min*1e3, grid.y_max*1e3],
                   cmap=THERMAL_CMAP, aspect="equal")
    fig.colorbar(im, ax=ax, label="Temperature (C)")
    ax.set_title(f"{title}  [z = {grid.z[z_index]*1e3:.1f} mm]")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig


def plot_slice_xz(grid, y_index=None, title="Temperature -- XZ Slice",
                  save_path=None):
    if y_index is None:
        y_index = grid.ny // 2
    T_slice = _C(grid.T[:, y_index, :])
    fig, ax = plt.subplots(figsize=(5, 9))
    im = ax.imshow(T_slice.T, origin="lower",
                   extent=[grid.x_min*1e3, grid.x_max*1e3,
                           grid.z_min*1e3, grid.z_max*1e3],
                   cmap=THERMAL_CMAP, aspect="auto")
    fig.colorbar(im, ax=ax, label="Temperature (C)")
    ax.set_title(f"{title}  [y = {grid.y[y_index]*1e3:.1f} mm]")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Temperature history
# ─────────────────────────────────────────────────────────────────────────────

def plot_history(solver, save_path=None):
    t      = np.array(solver.history_t)
    T_max  = np.array(solver.history_T_max)  - 273.15
    T_mean = np.array(solver.history_T_mean) - 273.15

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, T_max,  lw=2, color="#FF4444", label="T_max (battery)")
    ax.plot(t, T_mean, lw=2, color="#4488FF", label="T_mean (battery)", linestyle="--")
    ax.axhline(config.T_AMB - 273.15, color="grey", lw=1, linestyle=":", label="T_ambient")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (C)")
    ax.set_title("Battery Temperature vs Time")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Pack-specific plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_pack_top(grid, pack, z_index=None, save_path=None):
    """
    Top-down (XY) heat map for the full 2x2 pack with cell outlines annotated.
    Shows where heat is highest across the entire pack.
    """
    from core.pack import CELL_CENTRES, _R
    if z_index is None:
        z_index = grid.nz // 2

    T_slice = _C(grid.T[:, :, z_index])     # shape (nx, ny)
    vmin = T_slice.min()
    vmax = T_slice.max()

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(
        T_slice.T,
        origin="lower",
        extent=[grid.x_min*1e3, grid.x_max*1e3,
                grid.y_min*1e3, grid.y_max*1e3],
        cmap=THERMAL_CMAP,
        vmin=vmin, vmax=vmax,
        aspect="equal",
    )
    cb = fig.colorbar(im, ax=ax, label="Temperature (C)", fraction=0.046, pad=0.04)

    # Draw cell outlines + label hotspot
    temps = pack.cell_temperatures()
    hot_idx, hot_T = pack.hottest_cell()
    for i, (cx, cy) in enumerate(CELL_CENTRES):
        circle = plt.Circle((cx*1e3, cy*1e3), _R*1e3,
                             fill=False, edgecolor='white', linewidth=1.5, linestyle='--')
        ax.add_patch(circle)
        label_T = temps[i] - 273.15
        color   = "yellow" if i == hot_idx else "white"
        ax.text(cx*1e3, cy*1e3, f"Cell {i}\n{label_T:.1f}C",
                ha='center', va='center', fontsize=9, color=color,
                fontweight='bold')
        if i == hot_idx:
            # Hotspot star marker
            ax.plot(cx*1e3, cy*1e3, '*', markersize=16,
                    color='yellow', markeredgecolor='orange', markeredgewidth=1)

    ax.set_title(f"Pack Temperature Map (mid-height z={grid.z[z_index]*1e3:.0f}mm)\n"
                 f"Hotspot: Cell {hot_idx} at {hot_T-273.15:.1f}C   "
                 f"DOD={pack.ntgk.dod:.3f}")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig


def plot_cell_comparison(solver, pack_history: list[dict], save_path=None):
    """
    4-panel plot: T_max per cell vs time + NTGK metrics (DOD, voltage, Q).
    pack_history: list of stats() dicts saved every `save_every` steps.
    """
    if not pack_history:
        return None

    t    = [s["_t"]      for s in pack_history]
    dod  = [s["DOD"]     for s in pack_history]
    volt = [s["V_terminal_V"] for s in pack_history]
    Q_W  = [s["Q_per_cell_W"] for s in pack_history]
    T_cells = np.array([s["T_cells_C"] for s in pack_history])  # (nsnap, 4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["#FF4444", "#4488FF", "#44BB44", "#FF8800"]
    labels = [f"Cell {i}" for i in range(4)]

    # Temperature per cell
    ax = axes[0, 0]
    for i in range(4):
        ax.plot(t, T_cells[:, i], lw=2, color=colors[i], label=labels[i])
    ax.axhline(config.T_AMB - 273.15, color='grey', linestyle=':', lw=1, label='T_amb')
    ax.set_title("Max Temperature per Cell")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("T_max (C)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Hottest cell tracking
    ax = axes[0, 1]
    hot = [np.argmax(s["T_cells_C"]) for s in pack_history]
    ax.step(t, hot, where='mid', lw=2, color="#FF4444")
    ax.set_yticks([0,1,2,3]); ax.set_yticklabels(labels)
    ax.set_title("Hottest Cell Over Time")
    ax.set_xlabel("Time (s)"); ax.grid(True, alpha=0.3)

    # DOD and Voltage
    ax = axes[1, 0]
    ax2 = ax.twinx()
    l1, = ax.plot(t, dod,  lw=2, color="#8844FF", label="DOD")
    l2, = ax2.plot(t, volt, lw=2, color="#44BBBB", linestyle="--", label="V_terminal (V/cell)")
    ax.set_title("State of Discharge & Terminal Voltage")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DOD", color="#8844FF"); ax2.set_ylabel("V/cell", color="#44BBBB")
    ax.legend(handles=[l1, l2], fontsize=8)
    ax.grid(True, alpha=0.3)

    # Heat generation
    ax = axes[1, 1]
    ax.plot(t, Q_W, lw=2, color="#FF8800")
    ax.set_title("Heat Generation per Cell (NTGK)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Q (W)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("PyBatterySim -- 4S1P Pack NTGK Analysis", fontsize=13, fontweight='bold')
    fig.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150)
    return fig


def animate_pack_top(solver, pack_snapshots: list, fps=8, save_path=None):
    """Animate the top-down pack temperature field over saved snapshots."""
    if not pack_snapshots: return None
    from core.pack import CELL_CENTRES, _R
    grid = solver.grid

    all_T = np.stack([_C(s["T"][:, :, grid.nz // 2]) for s in pack_snapshots])
    vmin, vmax = all_T.min(), all_T.max()

    fig, ax = plt.subplots(figsize=(7, 7))
    im  = ax.imshow(all_T[0].T, origin="lower",
                    extent=[grid.x_min*1e3, grid.x_max*1e3,
                            grid.y_min*1e3, grid.y_max*1e3],
                    cmap=THERMAL_CMAP, vmin=vmin, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, label="Temperature (C)")
    for cx, cy in CELL_CENTRES:
        ax.add_patch(plt.Circle((cx*1e3, cy*1e3), _R*1e3,
                                fill=False, edgecolor='white', linewidth=1.5, linestyle='--'))
    ttl = ax.set_title("")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")

    def update(frame):
        s = pack_snapshots[frame]
        im.set_data(_C(s["T"][:, :, grid.nz // 2]).T)
        ttl.set_text(f"Pack Temperature  t={s['t']:.0f}s  DOD={s['dod']:.3f}")
        return im, ttl

    anim = animation.FuncAnimation(fig, update, frames=len(pack_snapshots),
                                   interval=1000/fps, blit=True)
    if save_path:
        ext = os.path.splitext(save_path)[-1].lower()
        anim.save(save_path, writer="pillow" if ext == ".gif" else None, fps=fps)
        print(f"[Viewer] Animation saved -> {save_path}")
    return anim


# ─────────────────────────────────────────────────────────────────────────────
# 3D PyVista rendering -- SURFACE mode (fast, no GPU transparency needed)
# ─────────────────────────────────────────────────────────────────────────────

def plot_3d(grid, pack=None, opacity=1.0):
    """
    Interactive 3-D surface rendering of the battery (pack).
    Uses extract_surface() instead of add_volume() -- 10-100x faster,
    works on any GPU including integrated graphics.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[Viewer] pyvista not available -- skipping 3D render.")
        return

    mesh = pv.ImageData()
    mesh.dimensions = (grid.nx+1, grid.ny+1, grid.nz+1)
    mesh.origin     = (grid.x_min, grid.y_min, grid.z_min)
    mesh.spacing    = (grid.dx,    grid.dy,    grid.dz)

    T_flat   = _C(grid.T).ravel(order="F")
    batt_flag = (grid.material_id == 1).ravel(order="F").astype(float)
    mesh.cell_data["Temperature_C"] = T_flat
    mesh.cell_data["is_battery"]    = batt_flag

    # Clip to battery cells only
    clipped = mesh.threshold(0.5, scalars="is_battery")

    # SURFACE extraction -- fast on any hardware
    surface = clipped.extract_surface()

    pl = pv.Plotter(window_size=(1000, 800))
    pl.add_mesh(
        surface,
        scalars="Temperature_C",
        cmap="coolwarm",
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Temperature (C)", "fmt": "%.1f"},
    )

    # Add hotspot annotation if pack provided
    if pack is not None:
        from core.pack import CELL_CENTRES, _R
        hot_idx, hot_T = pack.hottest_cell()
        cx, cy = CELL_CENTRES[hot_idx]
        pl.add_text(
            f"Hotspot: Cell {hot_idx}  {hot_T-273.15:.1f}C",
            position="upper_left",
            font_size=14,
            color="yellow",
        )

    pl.set_background("black")
    pl.show_axes()
    pl.show(title="PyBatterySim -- 3D Pack Temperature (Surface)")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Viewer class
# ─────────────────────────────────────────────────────────────────────────────

class Viewer:
    def __init__(self, solver, pack=None):
        self.solver = solver
        self.grid   = solver.grid
        self.pack   = pack

    def show_slice_xy(self, z_index=None, save=None):
        fn = plot_pack_top if self.pack else plot_slice_xy
        if self.pack:
            return plot_pack_top(self.grid, self.pack, z_index, save_path=save)
        return plot_slice_xy(self.grid, z_index, save_path=save)

    def show_slice_xz(self, y_index=None, save=None):
        return plot_slice_xz(self.grid, y_index, save_path=save)

    def show_history(self, save=None):
        return plot_history(self.solver, save_path=save)

    def show_pack_analysis(self, pack_history, save=None):
        return plot_cell_comparison(self.solver, pack_history, save_path=save)

    def animate(self, pack_snapshots=None, save=None, fps=8):
        if pack_snapshots:
            return animate_pack_top(self.solver, pack_snapshots, fps=fps, save_path=save)
        return None  # single-cell animation defined elsewhere if needed

    def show_3d(self):
        plot_3d(self.grid, pack=self.pack)
