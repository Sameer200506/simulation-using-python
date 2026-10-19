"""
core/grid.py — 3-D structured Cartesian voxel grid.

The grid owns the physical extent of the domain and the per-cell arrays:
    T           temperature field  [K]
    material_id integer label (0 = air, 1 = battery, …)
    rho, Cp, k_r, k_z  per-cell material properties (scalar arrays)
    Q           volumetric heat generation  [W/m³]

Cell centres are at  x[i], y[j], z[k]  with uniform spacing dx, dy, dz.
"""
import numpy as np
import config
from .material import Material, AIR


class Grid:
    """Structured 3-D Cartesian voxel grid."""

    def __init__(
        self,
        nx: int = config.NX,
        ny: int = config.NY,
        nz: int = config.NZ,
        x_extent: tuple[float, float] = None,
        y_extent: tuple[float, float] = None,
        z_extent: tuple[float, float] = None,
    ):
        # Domain extents default to a box slightly wider than the battery.
        half = config.RADIUS * config.DOMAIN_MARGIN
        self.x_min, self.x_max = x_extent or (-half, half)
        self.y_min, self.y_max = y_extent or (-half, half)
        self.z_min, self.z_max = z_extent or (0.0, config.HEIGHT)

        self.nx, self.ny, self.nz = nx, ny, nz

        # Cell spacings
        self.dx = (self.x_max - self.x_min) / nx
        self.dy = (self.y_max - self.y_min) / ny
        self.dz = (self.z_max - self.z_min) / nz

        # Cell-centre coordinate vectors
        self.x = np.linspace(self.x_min + 0.5 * self.dx,
                             self.x_max - 0.5 * self.dx, nx)
        self.y = np.linspace(self.y_min + 0.5 * self.dy,
                             self.y_max - 0.5 * self.dy, ny)
        self.z = np.linspace(self.z_min + 0.5 * self.dz,
                             self.z_max - 0.5 * self.dz, nz)

        # 3-D meshgrid of cell centres (used for geometry masking)
        self.XX, self.YY, self.ZZ = np.meshgrid(self.x, self.y, self.z,
                                                  indexing='ij')

        # ── Field arrays (shape: nx × ny × nz) ───────────────────────────────
        shape = (nx, ny, nz)
        self.T            = np.full(shape, config.T_INIT,  dtype=np.float64)
        self.material_id  = np.zeros(shape,                dtype=np.int32)
        self.rho          = np.full(shape, AIR.density,    dtype=np.float64)
        self.Cp           = np.full(shape, AIR.specific_heat, dtype=np.float64)
        self.k_r          = np.full(shape, AIR.conductivity,  dtype=np.float64)
        self.k_z          = np.full(shape, AIR.k_z,           dtype=np.float64)
        self.Q            = np.zeros(shape,                dtype=np.float64)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def set_material(self, mask: np.ndarray, mat: Material,
                     mat_id: int, heat_gen: float = 0.0) -> None:
        """Apply a material to all cells where mask is True."""
        self.material_id[mask] = mat_id
        self.rho[mask]         = mat.density
        self.Cp[mask]          = mat.specific_heat
        self.k_r[mask]         = mat.k_r
        self.k_z[mask]         = mat.k_z
        self.Q[mask]           = heat_gen

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def stats(self) -> dict:
        """Quick summary for logging."""
        batt_mask = self.material_id == 1
        return {
            "total_cells":   self.n_cells,
            "battery_cells": int(batt_mask.sum()),
            "air_cells":     int((~batt_mask).sum()),
            "T_min_K":       float(self.T.min()),
            "T_max_K":       float(self.T.max()),
            "T_mean_K":      float(self.T[batt_mask].mean()) if batt_mask.any() else float('nan'),
        }

    def __repr__(self) -> str:
        return (f"Grid({self.nx}×{self.ny}×{self.nz}, "
                f"dx={self.dx*1e3:.2f} mm, dy={self.dy*1e3:.2f} mm, "
                f"dz={self.dz*1e3:.2f} mm, "
                f"cells={self.n_cells:,})")
