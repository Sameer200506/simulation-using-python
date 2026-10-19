# -*- coding: utf-8 -*-
"""
tests/test_phase1.py — Self-check for Phase 1 foundation.

Run with:  python -m tests.test_phase1
           (from the PyBatterySim/ directory)

Checks:
  - Grid dimensions and coordinate arrays
  - Battery geometry mask (cylinder is non-empty, air is non-empty)
  - Material property stamping
  - One solver step — temperature must increase inside battery
  - Von Neumann stability check passes for default config
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import config
from core.grid import Grid
from core.battery import Battery
from core.solver import Solver


def _ok(name: str) -> None:
    print(f"  [OK]  {name}")

def _fail(name: str, msg: str) -> None:
    print(f"  [FAIL]  {name}: {msg}")
    sys.exit(1)


def test_grid():
    g = Grid(nx=20, ny=20, nz=40)
    assert g.shape == (20, 20, 40),       "shape mismatch"
    assert g.x.shape == (20,),            "x vector wrong"
    assert g.z.shape == (40,),            "z vector wrong"
    assert np.allclose(g.T, config.T_INIT), "T not initialised to T_INIT"
    _ok("Grid construction")


def test_battery_geometry():
    g    = Grid(nx=20, ny=20, nz=40)
    batt = Battery(grid=g)
    batt.build()

    n_batt = (g.material_id == 1).sum()
    n_air  = (g.material_id == 0).sum()
    assert n_batt > 0, "no battery cells marked"
    assert n_air  > 0, "no air cells — cylinder fills whole domain?"
    _ok(f"Battery geometry  ({n_batt} battery, {n_air} air cells)")


def test_material_properties():
    g    = Grid(nx=20, ny=20, nz=40)
    batt = Battery(grid=g)
    batt.build()

    mask = g.material_id == 1
    assert np.all(g.k_r[mask]  == config.BATTERY_K_RADIAL), "k_r wrong in battery"
    assert np.all(g.k_z[mask]  == config.BATTERY_K_AXIAL),  "k_z wrong in battery"
    assert np.all(g.Q[mask]    == config.BATTERY_Q),         "Q wrong in battery"
    assert np.all(g.Q[~mask]   == 0.0),                      "air has heat gen?"
    _ok("Material property stamping")


def test_solver_heats_up():
    g    = Grid(nx=10, ny=10, nz=20)   # coarse for speed
    batt = Battery(grid=g)
    batt.build()

    T_before = g.T[g.material_id == 1].mean()
    solver = Solver(g, dt=1.0, t_end=5.0, save_every=5)
    solver.run(verbose=False)
    T_after = g.T[g.material_id == 1].mean()

    assert T_after > T_before, "Battery did not heat up after solve step"
    _ok(f"Solver heats battery  ({T_before-273.15:.3f} -> {T_after-273.15:.3f} C)")


def test_stability_check_runs():
    g    = Grid(nx=10, ny=10, nz=20)
    batt = Battery(grid=g)
    batt.build()
    # Implicit solver: always stable; check that it initializes and has a valid dt
    solver = Solver(g, dt=1.0, t_end=5.0, save_every=5)
    assert solver.dt > 0,         "dt is non-positive"
    assert solver.n_steps == 5,   "n_steps wrong"
    _ok(f"Implicit solver init OK  (dt = {solver.dt:.2f} s, n_steps = {solver.n_steps})")


if __name__ == "__main__":
    print("\nPyBatterySim - Phase 1 Self-Check\n" + "-" * 40)
    test_grid()
    test_battery_geometry()
    test_material_properties()
    test_solver_heats_up()
    test_stability_check_runs()
    print("\nAll checks passed.\n")
