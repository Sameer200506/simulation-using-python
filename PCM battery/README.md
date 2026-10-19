# PCM Battery Pack – Transient Thermal Simulation

Simulates the transient heat conduction in a **4-cell 26650 Li-ion pack**
surrounded by a **Phase Change Material (PCM)** block using the finite element
method (FEniCSx / DOLFINx + gmsh).

---

## Physics

| Quantity | Model |
|---|---|
| PDE | Transient heat conduction: ρ c_p_eff ∂T/∂t = ∇·(k∇T) + q̇ |
| Time discretisation | Implicit Backward-Euler (BDF-1) |
| Spatial discretisation | Continuous Galerkin P1 (CG1) triangles |
| PCM phase change | Apparent heat capacity method: c_p_eff(T) = c_p + L ∂f_l/∂T |
| Outer boundary | Robin: −k ∇T·n̂ = h(T − T_∞), h = 5 W/(m²·K) |

## Geometry

```
 ┌────────────────────────┐
 │  PCM  (100×100 mm)     │
 │   ○ B3       ○ B4      │
 │   ○ B1       ○ B2      │
 └────────────────────────┘
```

- 4 × 26650 batteries (r = 13 mm) in a 2×2 grid, 30 mm pitch
- PCM block: 100 mm × 100 mm
- 2-D cross-section (per unit axial depth)

## Key parameters

| Parameter | Value |
|---|---|
| Battery heat generation q̇ | 60 000 W/m³ (≈ 1C) |
| PCM (RT-42) solidus / liquidus | 40 °C / 44 °C |
| PCM latent heat L | 165 kJ/kg |
| Ambient temperature T_∞ | 25 °C |
| Default time step Δt | 30 s |
| Default simulation time | 3600 s (1 h) |

---

## Installation

> **Windows users**: FEniCSx does not run natively on Windows.
> Use **WSL2** (Ubuntu) or **Docker** (see below).

### Option A – WSL2 / Linux / macOS (Conda)

```bash
# 1. Install Miniforge or Anaconda
# 2. Create the environment
conda env create -f environment.yml
conda activate fenics-pcm

# 3. Run
python main.py
```

### Option B – Docker (any OS, no Conda needed)

```bash
docker run --rm -it \
  -v "$(pwd)":/sim -w /sim \
  dolfinx/dolfinx:v0.8.0 \
  bash -c "pip install gmsh matplotlib scipy && python main.py"
```

---

## Usage

```
python main.py [OPTIONS]

Options:
  --dt        DT          Time step [s]                  (default: 30)
  --t_end     T_END       Total simulation time [s]      (default: 3600)
  --output    DIR         Output directory               (default: output/)
  --mesh_bat  SIZE        Battery mesh size [m]          (default: 0.0015)
  --mesh_pcm  SIZE        PCM mesh size [m]              (default: 0.004)
  --save_every N          XDMF snapshot interval         (default: 5)
  --help                  Show this message and exit
```

Example – quick 30-minute run with 15 s steps:

```bash
python main.py --dt 15 --t_end 1800 --output quick_run
```

---

## Output structure

```
output/
├── mesh/
│   ├── mesh.xdmf             ← Open in ParaView to see the mesh + tags
│   └── mesh.h5
├── pcm/
│   ├── temperature.xdmf      ← Temperature snapshots WITH PCM
│   └── temperature.h5
├── nopcm/
│   ├── temperature_nopcm.xdmf  ← Temperature snapshots WITHOUT PCM (air)
│   └── temperature_nopcm.h5
└── results/
    ├── fig1_battery_temperature.png
    ├── fig2_pcm_temperature.png
    ├── fig3_max_battery_temp.png
    ├── fig4_heat_absorbed.png
    ├── fig5_liquid_fraction.png
    └── results.csv
```

### CSV columns

| Column | Unit | Description |
|---|---|---|
| time_s | s | Simulation time |
| T_bat_max_PCM_degC | °C | Max battery temperature (with PCM) |
| T_bat_avg_PCM_degC | °C | Avg battery temperature (with PCM) |
| T_pcm_avg_degC | °C | Avg PCM temperature |
| T_bat_max_noPCM_degC | °C | Max battery temperature (no PCM) |
| T_bat_avg_noPCM_degC | °C | Avg battery temperature (no PCM) |
| delta_T_bat_max_K | K | Temperature reduction due to PCM |
| liquid_fraction | – | Volume-averaged PCM liquid fraction |
| Q_latent_kJ_per_m | kJ/m | Latent heat stored in PCM |
| Q_sensible_kJ_per_m | kJ/m | Sensible heat stored in PCM |
| Q_total_PCM_kJ_per_m | kJ/m | Total heat absorbed by PCM |
| E_generated_kJ_per_m | kJ/m | Cumulative battery energy generated |

---

## File structure

```
PCM battery/
├── geometry.py     — gmsh geometry builder (OCC kernel, 2-D cross-section)
├── mesh.py         — mesh generation + DOLFINx conversion + XDMF export
├── materials.py    — thermophysical property functions (battery, PCM, air)
├── solver.py       — BatteryPCMSolver + TimeSeries (backward-Euler FEM)
├── postprocess.py  — PostProcessor (5 plots, CSV, summary)
├── main.py         — entry point + CLI
└── environment.yml — Conda environment specification
```

---

## ParaView visualisation

1. Open `output/pcm/temperature.xdmf` in ParaView.
2. Apply a **Warp by Scalar** or colour map on `Temperature`.
3. Use **Play** to animate the transient solution.
4. Open `output/mesh/mesh.xdmf` to see the subdomain tags (colour by `subdomains`).
