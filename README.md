# simulation using python (Li-ion battery pack simulations)

Two Python simulation projects for lithium-ion battery packs:

## 1. `PyBatterySim/` — NTGK electrochemical + thermal pack model

A pure-Python battery pack simulator built around the **NTGK** (Newman–
Tiedemann–Gu–Kim) model with a finite-volume thermal solve.

- Cell electrical model + coupled heat generation (`core/ntgk.py`,
  `core/battery.py`).
- 1D/3D grid and material handling (`core/grid.py`, `core/material.py`,
  `core/geometry.py`).
- Pack assembly and busbar variants (`core/pack.py`, `core/pack_busbars.py`,
  `main_pack.py`, `main_busbars.py`).
- Sparse solvers, VT comparison plots and Excel-driven material data
  (`core/solver.py`, `plot_vt_comparison.py`, `LIB.xlsx`).

```bash
cd PyBatterySim
pip install -r requirements.txt
python main.py
python main_pack.py
```

## 2. `PCM battery/` — transient thermal simulation with phase-change material

A 4-cell 26650 Li-ion pack embedded in a PCM block, solved with **FEniCSx /
DOLFINx + Gmsh**.

- Transient heat conduction with implicit backward Euler.
- Apparent heat-capacity phase change, Robin outer boundary
  (h = 5 W/m²K).
- Modules: `geometry.py`, `mesh.py`, `materials.py`, `solver.py`,
  `postprocess.py`, `main.py`.

```bash
conda env create -f "PCM battery/environment.yml"
conda activate pcm-battery          # or the name in environment.yml
cd "PCM battery"
python main.py
```

Outputs are written to `output/` (VTU/PVD + temperature plots).

## Notes

- The bundled `PCM battery/C?Usersmoham/...` Miniforge environment is a local
  conda install and is **not** part of the source; it is excluded from this
  repository.
- `__pycache__`, solver outputs and other generated artefacts are ignored.
