"""
config.py — PyBatterySim global configuration
All physical constants and simulation defaults live here.
SI units throughout.
"""

# ─── Battery geometry ────────────────────────────────────────────────────────
RADIUS = 0.013          # m  (26 mm diameter → 13 mm radius)
HEIGHT = 0.065          # m  (65 mm)

# ─── Computational domain ────────────────────────────────────────────────────
# Box slightly larger than the battery so the outer air/boundary is captured.
DOMAIN_MARGIN = 1.5     # multiplier on the battery radius for x/y extent
# NX * NY should resolve the cylinder well; NZ scales with aspect ratio.
# 30x30x60 = 54,000 cells -- well-resolved for demo/iteration (use 40x40x80 for production)
NX = 30
NY = 30
NZ = 60

# ─── Time integration ────────────────────────────────────────────────────────
T_END   = 1800.0        # s   (simulate 1800 s)
DT      = 1.0           # s   time step
SAVE_EVERY = 60         # steps between saved snapshots

# ─── Initial / boundary conditions ───────────────────────────────────────────
T_INIT  = 305.65        # K   (32.5 °C to match Excel starting temperature)
T_AMB   = 305.65        # K   ambient temperature
H_CONV  = 25.0          # W/(m²·K)  calibrated convection coefficient (targets ~66.7C peak)

# ─── Material defaults (26650 NMC cell, approximate literature values) ────────
BATTERY_DENSITY      = 2500.0   # kg/m³
BATTERY_CP           = 1000.0   # J/(kg·K)
BATTERY_K_RADIAL     = 0.8      # W/(m·K)  radial (poor — through separator/jelly-roll)
BATTERY_K_AXIAL      = 25.0     # W/(m·K)  axial  (good — along tabs)
BATTERY_Q            = 50000.0  # W/m³  constant volumetric heat generation

# ─── Ambient "air" properties (cells outside cylinder) ───────────────────────
AIR_DENSITY = 1.2       # kg/m³
AIR_CP      = 1005.0    # J/(kg·K)
# ponytail: AIR_K set to 0.15 to act as an effective convective conductivity.
# Calibrated against experimental dataset (LIB.xlsx Sheet2).
AIR_K       = 0.15      # W/(m·K) (effective convective thermal conductivity)

# ─── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
