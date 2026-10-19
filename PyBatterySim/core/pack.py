"""
core/pack.py -- 4-battery 2x2 pack geometry and thermal coupling.

Layout (top-down view, 4S1P — 4 cells in series, 1 in parallel):

    Gap = 3 mm between cell surfaces
    Cell centres at (+/- (R + gap/2)) in both x and y

    (-cx, +cy)  (+cx, +cy)       cell 0   cell 1
    (-cx, -cy)  (+cx, -cy)       cell 2   cell 3

    cx = cy = R + gap/2 = 13 + 1.5 = 14.5 mm

The pack grid is larger than a single-cell grid; it contains all 4 cylinders
plus surrounding air for natural convection.

Usage
-----
    from core.pack import Pack
    pack = Pack(ntgk_model, I_discharge=10.7)
    pack.build()          # stamps all 4 batteries onto the grid
    pack.update_Q()       # update heat source from NTGK model
    pack.advance(dt)      # advance DOD
"""
from __future__ import annotations
import numpy as np
import config
from .grid import Grid
from .material import BATTERY, AIR
from .geometry import mark_cylinder
from .ntgk import NTGKModel


# Material IDs
MAT_AIR     = 0
MAT_BATTERY = 1   # cells 0-3 share the same material ID
                  # (temperature is per-voxel, ID is for material lookup)

# Pack geometry (SI units)
_R    = config.RADIUS        # 13 mm
_H    = config.HEIGHT        # 65 mm
_GAP  = 0.003                # 3 mm between cell surfaces
_CX   = _R + _GAP / 2.0     # 14.5 mm  -- cell centre x/y offset
_MARGIN = 0.008              # 8 mm air margin around pack

# Cell centres (x, y) for the 2x2 arrangement
CELL_CENTRES: list[tuple[float, float]] = [
    (-_CX, +_CX),   # cell 0 — top-left
    (+_CX, +_CX),   # cell 1 — top-right
    (-_CX, -_CX),   # cell 2 — bottom-left
    (+_CX, -_CX),   # cell 3 — bottom-right
]

# Pack domain extents
_HALF_DOMAIN = _CX + _R + _MARGIN   # ~35.5 mm


class Pack:
    """
    2x2 (4S1P) battery pack.

    Parameters
    ----------
    ntgk      : NTGKModel instance (shared parameters for all cells —
                they are assumed identical 26650 cells)
    I         : discharge current [A]
    nx, ny, nz: grid resolution
    """

    def __init__(
        self,
        ntgk: NTGKModel,
        I: float    = None,
        nx: int     = 50,
        ny: int     = 50,
        nz: int     = 50,
    ):
        self.ntgk = ntgk
        self.I    = I if I is not None else ntgk._I_estimated
        self.n_cells = 4

        # Build grid sized around the whole pack
        half = _HALF_DOMAIN
        self.grid = Grid(
            nx=nx, ny=ny, nz=nz,
            x_extent=(-half, half),
            y_extent=(-half, half),
            z_extent=(0.0, _H),
        )

        # Per-battery masks — computed during build()
        self._masks: list[np.ndarray] = []

    # ── Geometry stamping ─────────────────────────────────────────────────────

    def build(self) -> Grid:
        """Stamp all 4 cylinders onto the grid. Returns the populated Grid."""
        g = self.grid
        self._masks = []

        for cx, cy in CELL_CENTRES:
            mask = mark_cylinder(g, radius=_R, z_bottom=0.0, z_top=_H,
                                 cx=cx, cy=cy)
            g.set_material(mask, BATTERY, mat_id=MAT_BATTERY,
                           heat_gen=0.0)     # Q set by update_Q()
            self._masks.append(mask)

        # Combined battery mask (any cell)
        self._all_battery = np.logical_or.reduce(self._masks)

        # Initial heat generation
        self.update_Q(self.ntgk.dod)
        return g

    # ── NTGK coupling ─────────────────────────────────────────────────────────

    def cell_avg_temperature(self, cell_idx: int) -> float:
        """Average temperature [K] of cell `cell_idx` from the grid."""
        return float(self.grid.T[self._masks[cell_idx]].mean())

    def update_Q(self, dod: float | None = None) -> None:
        """
        Recompute volumetric heat generation for each battery cell.
        Called once per time step before the thermal solve.
        """
        g   = self.grid
        dod = dod if dod is not None else self.ntgk.dod
        for i, mask in enumerate(self._masks):
            T_avg = self.cell_avg_temperature(i) if mask.any() else (config.T_INIT)
            q_vol = self.ntgk.heat_per_m3(self.I, T_avg, dod)
            g.Q[mask] = q_vol

    def advance(self, dt: float) -> None:
        """Advance NTGK DOD by one time step."""
        self.ntgk.advance(self.I, dt)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def cell_temperatures(self) -> list[float]:
        """Return [T_max_K] for each of the 4 cells."""
        return [float(self.grid.T[m].max()) for m in self._masks]

    def hottest_cell(self) -> tuple[int, float]:
        """Return (cell_index, T_max_K) for the hottest cell."""
        temps = self.cell_temperatures()
        idx   = int(np.argmax(temps))
        return idx, temps[idx]

    def stats(self) -> dict:
        temps = self.cell_temperatures()
        return {
            "DOD":          self.ntgk.dod,
            "U_OCV_V":      self.ntgk.U_ocv(),
            "V_terminal_V": self.ntgk.V_terminal(self.I),
            "Q_per_cell_W": self.ntgk.heat_per_cell(self.I, temps[0] if temps else config.T_INIT),
            "T_cells_C":    [t - 273.15 for t in temps],
            "T_max_C":      max(temps) - 273.15,
            "hottest_cell": int(np.argmax(temps)),
        }

    def __repr__(self) -> str:
        return (f"Pack(4S1P, I={self.I:.1f}A, "
                f"grid={self.grid.nx}x{self.grid.ny}x{self.grid.nz}, "
                f"DOD={self.ntgk.dod:.3f})")
