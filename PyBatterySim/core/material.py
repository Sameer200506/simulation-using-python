"""
core/material.py — Material property definitions.

A Material is just a named bundle of thermophysical properties.
Each voxel in the grid holds one Material reference.
"""
from dataclasses import dataclass
import config


@dataclass(frozen=True)
class Material:
    """Immutable set of thermophysical properties for one material."""
    name: str
    density: float          # kg/m³
    specific_heat: float    # J/(kg·K)
    conductivity: float     # W/(m·K)  isotropic default
    conductivity_axial: float | None = None  # W/(m·K) if anisotropic (None → use conductivity)

    @property
    def k_r(self) -> float:
        """Radial (in-plane) thermal conductivity."""
        return self.conductivity

    @property
    def k_z(self) -> float:
        """Axial thermal conductivity (falls back to isotropic k if not set)."""
        return self.conductivity_axial if self.conductivity_axial is not None else self.conductivity

    @property
    def alpha(self) -> float:
        """Thermal diffusivity  α = k / (ρ Cp)  [m²/s]."""
        return self.conductivity / (self.density * self.specific_heat)


# ─── Pre-built material catalogue ────────────────────────────────────────────

BATTERY = Material(
    name="battery_cell",
    density=config.BATTERY_DENSITY,
    specific_heat=config.BATTERY_CP,
    conductivity=config.BATTERY_K_RADIAL,
    conductivity_axial=config.BATTERY_K_AXIAL,
)

AIR = Material(
    name="air",
    density=config.AIR_DENSITY,
    specific_heat=config.AIR_CP,
    conductivity=config.AIR_K,
)

TERMINAL = Material(
    name="terminal",
    density=8908.0,          # kg/m³   — nickel-plated steel cap
    specific_heat=444.0,     # J/(kg·K)
    conductivity=91.0,       # W/(m·K) — nickel (isotropic)
)

BUSBAR = Material(
    name="busbar",
    density=8960.0,          # kg/m³   — copper strip
    specific_heat=385.0,     # J/(kg·K)
    conductivity=400.0,      # W/(m·K) — copper (isotropic)
)

MATERIAL_CATALOGUE: dict[str, Material] = {
    m.name: m for m in (BATTERY, AIR, TERMINAL, BUSBAR)
}
