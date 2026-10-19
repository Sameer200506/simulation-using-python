"""
core/battery.py — High-level battery object.

Battery.build() stamps the battery geometry and material onto a Grid,
returning the populated grid ready for the solver.

This is the single place that knows about the physical structure of a
26650 cell.  Later we'll add sub-regions (jelly roll, tabs, casing) here.
"""
import numpy as np
import config
from .grid import Grid
from .material import BATTERY, AIR
from .geometry import mark_cylinder


# Material IDs — integer labels stored in grid.material_id
MAT_AIR     = 0
MAT_BATTERY = 1


class Battery:
    """
    Represents a single cylindrical 26650 lithium-ion cell.

    Parameters
    ----------
    radius : float  [m]   outer radius of the cell
    height : float  [m]   total height of the cell
    heat_gen : float [W/m³]  volumetric heat generation (constant for now)
    grid   : Grid | None   pass an existing grid or let Battery create one
    """

    def __init__(
        self,
        radius: float = config.RADIUS,
        height: float = config.HEIGHT,
        heat_gen: float = config.BATTERY_Q,
        grid: Grid | None = None,
    ):
        self.radius   = radius
        self.height   = height
        self.heat_gen = heat_gen
        self.grid     = grid or Grid()

    # ── Public ────────────────────────────────────────────────────────────────

    def build(self) -> Grid:
        """
        Stamp the battery geometry onto the grid.

        Returns the populated Grid so callers can chain:
            grid = Battery().build()
        """
        grid = self.grid

        # Everything is already initialised to AIR in Grid.__init__.
        # We only need to mark the battery cells.
        mask = mark_cylinder(grid, radius=self.radius,
                             z_bottom=0.0, z_top=self.height)
        grid.set_material(mask, BATTERY, mat_id=MAT_BATTERY,
                          heat_gen=self.heat_gen)

        self._mask = mask          # keep for quick access by solver/viewer
        return grid

    @property
    def mask(self) -> np.ndarray:
        """Boolean mask of battery cells (available after build())."""
        if not hasattr(self, '_mask'):
            raise RuntimeError("Call Battery.build() before accessing mask.")
        return self._mask

    def info(self) -> dict:
        """Return a dict of physical properties for logging / GUI display."""
        return {
            "radius_mm":    self.radius  * 1e3,
            "height_mm":    self.height  * 1e3,
            "heat_gen_W_m3": self.heat_gen,
            "volume_m3":    np.pi * self.radius**2 * self.height,
            "total_heat_W": np.pi * self.radius**2 * self.height * self.heat_gen,
        }

    def __repr__(self) -> str:
        return (f"Battery(r={self.radius*1e3:.1f} mm, "
                f"h={self.height*1e3:.1f} mm, "
                f"Q={self.heat_gen:.0f} W/m³)")
