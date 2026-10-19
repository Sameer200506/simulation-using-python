"""
core/pack_busbars.py  -- 4-cell 2x2 pack WITH busbars and terminals.

Physical geometry
-----------------
Each 26650 cell has:
  - A nickel terminal disk  (1 mm thick) at the bottom (z=0  to z=1mm)
  - A nickel terminal disk  (1 mm thick) at the top    (z=H-1mm to z=H)

Three copper busbars connect the cells in 4S series:
  Busbar 1 (at z = H+1mm level): Cell 0 top  <-> Cell 1 top
  Busbar 2 (at z = -1mm  level): Cell 1 bottom <-> Cell 2 bottom
  Busbar 3 (at z = H+1mm level): Cell 2 top  <-> Cell 3 top

Pack (+) lead at Cell 0 top, Pack (-) lead at Cell 3 bottom.

Heat generation (pure physics, no Excel)
-----------------------------------------
Battery cell  : Q_cell  = I²·R_int + |I·T·α_entropy|     (NTGK pre-fitted)
Terminal cap  : Q_term  = I²·R_contact / V_terminal        (contact resistance)
Busbar        : Q_bus   = I²·R_busbar  / V_busbar          (bulk resistance ρ·L/A)

  R_contact  = 2 mΩ per terminal weld (typical laser/ultrasonic weld)
  R_busbar   = ρ_cu · L / A   (copper: ρ=1.72e-8 Ω·m)
  Busbar dims: 2 mm thick, 8 mm wide, ~29 mm long

Material IDs
------------
  0  air
  1  battery cell
  2  terminal (nickel)
  3  busbar   (copper)
"""
from __future__ import annotations
import numpy as np
import config
from .grid      import Grid
from .material  import BATTERY, AIR, TERMINAL, BUSBAR
from .geometry  import mark_cylinder, mark_box
from .ntgk      import NTGKModel

# ── Pack geometry constants ───────────────────────────────────────────────────
_R        = config.RADIUS        # 13 mm
_H        = config.HEIGHT        # 65 mm
_GAP      = 0.003                # 3 mm between cell surfaces
_CX       = _R + _GAP / 2.0     # 14.5 mm  — cell centre offset
_MARGIN   = 0.010                # outer air margin

_TERMINAL_T  = 0.001            # 1 mm  terminal disk thickness
_BUSBAR_T    = 0.002            # 2 mm  busbar thickness
_BUSBAR_W    = 0.008            # 8 mm  busbar width (narrow strip)
_BUSBAR_GAP  = 0.0              # 0 mm  gap (welded directly to terminals)

# Domain extents: extend z to accommodate busbars above and below
_Z_BOT    = -(_TERMINAL_T + _BUSBAR_T + _BUSBAR_GAP)  # ~ -4 mm
_Z_TOP    = _H + _TERMINAL_T + _BUSBAR_T + _BUSBAR_GAP # ~ +69 mm
_HALF_XY  = _CX + _R + _MARGIN  # ~ 36 mm

# Material IDs
MAT_AIR      = 0
MAT_BATTERY  = 1
MAT_TERMINAL = 2
MAT_BUSBAR   = 3

# Cell centres [x, y]
CELL_CENTRES: list[tuple[float, float]] = [
    (-_CX, +_CX),   # cell 0 — top-left
    (+_CX, +_CX),   # cell 1 — top-right
    (-_CX, -_CX),   # cell 2 — bottom-left
    (+_CX, -_CX),   # cell 3 — bottom-right
]

# Electrical parameters (pure physics, no Excel)
_RHO_COPPER    = 1.72e-8    # Ω·m  resistivity of copper
_R_CONTACT     = 0.002      # Ω    contact resistance per terminal weld (2 mΩ)
_ENTROPY_COEFF = -3.0e-4    # V/K  LiB entropic coefficient


class PackWithBusbars:
    """
    2x2 4S1P pack with terminal caps and copper busbars.

    Parameters
    ----------
    ntgk : NTGKModel  — pre-fitted NTGK (no Excel required by default)
    I    : discharge current [A]
    nx, ny, nz : grid resolution
    """

    def __init__(
        self,
        ntgk: NTGKModel,
        I:    float = None,
        nx:   int   = 50,
        ny:   int   = 50,
        nz:   int   = 60,         # extra cells for busbar layers
    ):
        self.ntgk    = ntgk
        self.I       = I if I is not None else ntgk._I_estimated
        self.n_cells = 4

        self.grid = Grid(
            nx=nx, ny=ny, nz=nz,
            x_extent=(-_HALF_XY, +_HALF_XY),
            y_extent=(-_HALF_XY, +_HALF_XY),
            z_extent=(_Z_BOT,    _Z_TOP),
        )

        # Masks filled during build()
        self._cell_masks:     list[np.ndarray] = []
        self._terminal_masks: list[np.ndarray] = []  # [bot0,top0, bot1,top1, ...]
        self._busbar_masks:   list[np.ndarray] = []
        self._busbar_R:       list[float]      = []  # Ω each busbar

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> Grid:
        g = self.grid

        # 1. Stamp battery cells
        for cx, cy in CELL_CENTRES:
            mask = mark_cylinder(g, radius=_R,
                                 z_bottom=0.0, z_top=_H, cx=cx, cy=cy)
            g.set_material(mask, BATTERY, mat_id=MAT_BATTERY, heat_gen=0.0)
            self._cell_masks.append(mask)

        # 2. Stamp terminal caps (nickel disks, overwrite outer battery cells)
        t_thick = max(_TERMINAL_T, g.dz)
        for cx, cy in CELL_CENTRES:
            # bottom terminal
            bot = mark_cylinder(g, radius=_R,
                                z_bottom=0.0,
                                z_top=t_thick,
                                cx=cx, cy=cy)
            g.set_material(bot, TERMINAL, mat_id=MAT_TERMINAL, heat_gen=0.0)
            # top terminal
            top = mark_cylinder(g, radius=_R,
                                z_bottom=_H - t_thick,
                                z_top=_H,
                                cx=cx, cy=cy)
            g.set_material(top, TERMINAL, mat_id=MAT_TERMINAL, heat_gen=0.0)
            self._terminal_masks += [bot, top]

        # 3. Stamp busbars (copper rectangular strips)
        #    Series wiring: Cell0-top  <-B1-> Cell1-top  (at z = H + gap)
        #                   Cell1-bot  <-B2-> Cell2-bot  (at z = -(gap))
        #                   Cell2-top  <-B3-> Cell3-top  (at z = H + gap)
        cx0, cy0 = CELL_CENTRES[0]   # (-CX, +CX)
        cx1, cy1 = CELL_CENTRES[1]   # (+CX, +CX)
        cx2, cy2 = CELL_CENTRES[2]   # (-CX, -CX)
        cx3, cy3 = CELL_CENTRES[3]   # (+CX, -CX)

        busbar_specs = [
            # (x_lo, x_hi, y_lo, y_hi, z_lo, z_hi, length)
            # B1: top of cells 0 & 1, horizontal strip at y=+CX row
            (min(cx0, cx1) - _R, max(cx0, cx1) + _R,
             cy0 - _BUSBAR_W/2, cy0 + _BUSBAR_W/2,
             _H + _BUSBAR_GAP,  _H + _BUSBAR_GAP + _BUSBAR_T,
             abs(cx1 - cx0) + 2*_R),

            # B2: bottom of cells 1 & 2, vertical strip at x=+CX column
            (cx1 - _BUSBAR_W/2, cx1 + _BUSBAR_W/2,
             min(cy1, cy3) - _R, max(cy1, cy3) + _R,
             _Z_BOT,             _Z_BOT + _BUSBAR_T,
             abs(cy1 - cy3) + 2*_R),

            # B3: top of cells 2 & 3, horizontal strip at y=-CX row
            (min(cx2, cx3) - _R, max(cx2, cx3) + _R,
             cy2 - _BUSBAR_W/2, cy2 + _BUSBAR_W/2,
             _H + _BUSBAR_GAP,  _H + _BUSBAR_GAP + _BUSBAR_T,
             abs(cx3 - cx2) + 2*_R),
        ]

        for x0, x1, y0, y1, z0, z1, L in busbar_specs:
            mask = mark_box(g, x0, x1, y0, y1, z0, z1)
            g.set_material(mask, BUSBAR, mat_id=MAT_BUSBAR, heat_gen=0.0)
            self._busbar_masks.append(mask)
            A = _BUSBAR_T * _BUSBAR_W
            self._busbar_R.append(_RHO_COPPER * L / A)

        print(f"[Pack] Busbars: {len(self._busbar_masks)}")
        for i, R in enumerate(self._busbar_R):
            Q = self.I ** 2 * R
            print(f"  Busbar {i}: R = {R*1e6:.1f} uOhm  Q = {Q*1e3:.2f} mW "
                  f"(copper strip {_BUSBAR_T*1e3:.0f}mm x {_BUSBAR_W*1e3:.0f}mm)")
        Q_contact = self.I ** 2 * _R_CONTACT
        print(f"[Pack] Terminal contact: R = {_R_CONTACT*1e3:.1f} mOhm  "
              f"Q = {Q_contact:.3f} W per weld")

        # Initial heat generation
        self.update_Q()
        return g

    # ── NTGK + resistive heat coupling ────────────────────────────────────────

    def cell_avg_temperature(self, cell_idx: int) -> float:
        return float(self.grid.T[self._cell_masks[cell_idx]].mean())

    def update_Q(self) -> None:
        """
        Refresh all volumetric heat sources:
          Battery cells  : NTGK (Joule + entropic)
          Terminal caps  : I² · R_contact  (contact resistance)
          Busbars        : I² · R_busbar   (copper bulk resistance)
        """
        g   = self.grid
        dod = self.ntgk.dod
        I   = self.I

        # Battery cells
        for i, mask in enumerate(self._cell_masks):
            T_avg = self.cell_avg_temperature(i)
            q_vol = self.ntgk.heat_per_m3(I, T_avg, dod)
            g.Q[mask] = q_vol

        # Terminal caps (nickel, contact resistance dominated)
        Q_contact_total = I ** 2 * _R_CONTACT          # W per terminal
        for mask in self._terminal_masks:
            vol = float(mask.sum()) * g.cell_volume
            if vol > 0:
                g.Q[mask] = Q_contact_total / vol      # W/m³

        # Busbars (copper bulk resistance)
        for mask, R_bus in zip(self._busbar_masks, self._busbar_R):
            Q_bus = I ** 2 * R_bus                     # W per busbar
            vol   = float(mask.sum()) * g.cell_volume
            if vol > 0:
                g.Q[mask] = Q_bus / vol                # W/m³

    def advance(self, dt: float) -> None:
        self.ntgk.advance(self.I, dt)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def cell_temperatures(self) -> list[float]:
        return [float(self.grid.T[m].max()) for m in self._cell_masks]

    def hottest_component(self) -> tuple[str, float]:
        """Return (component_name, T_max_C) for the hottest voxel in the pack."""
        T = self.grid.T
        component_masks = {
            "battery": np.logical_or.reduce(self._cell_masks),
            "terminal": np.logical_or.reduce(self._terminal_masks) if self._terminal_masks else None,
            "busbar":   np.logical_or.reduce(self._busbar_masks)   if self._busbar_masks   else None,
        }
        best_name, best_T = "battery", -1e9
        for name, mask in component_masks.items():
            if mask is not None and mask.any():
                t = float(T[mask].max()) - 273.15
                if t > best_T:
                    best_T, best_name = t, name
        return best_name, best_T

    def stats(self) -> dict:
        temps = self.cell_temperatures()
        comp, comp_T = self.hottest_component()
        Q_cell = self.ntgk.heat_per_cell(self.I, temps[0] if temps else config.T_INIT)
        Q_contact = self.I ** 2 * _R_CONTACT
        Q_busbars = sum(self.I ** 2 * R for R in self._busbar_R)
        return {
            "DOD":              self.ntgk.dod,
            "V_terminal_V":     self.ntgk.V_terminal(self.I),
            "Q_cell_W":         Q_cell,
            "Q_contact_W":      Q_contact,
            "Q_busbar_total_W": Q_busbars,
            "Q_total_W":        Q_cell * 4 + Q_contact * 8 + Q_busbars,
            "T_cells_C":        [t - 273.15 for t in temps],
            "T_max_C":          max(temps) - 273.15,
            "hottest_cell":     int(np.argmax(temps)),   # solver compat
            "hottest_component": comp,
            "hottest_T_C":       comp_T,
        }
