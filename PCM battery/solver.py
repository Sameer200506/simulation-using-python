"""
solver.py
---------
BatteryPCMSolver – backward-Euler FEM solver for transient heat conduction.

Discretisation
──────────────
  Spatial  : Continuous Galerkin P1 (CG1) on the mesh/submesh.
  Temporal : Implicit Backward-Euler (BDF-1).

Modes
─────
  1. WITH PCM (use_pcm=True):
     Solves on the full composite 2-D mesh.
     PCM properties are temperature-dependent and updated at each step.
     Robin boundary condition is applied on the outer block surface.

  2. WITHOUT PCM (use_pcm=False):
     Solves ONLY on the batteries by extracting a submesh.
     Robin boundary condition (natural convection to ambient) is applied
     directly on the outer cylindrical surface of the battery cells.
     This models the physical "pack in air" setup correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from mpi4py import MPI

import dolfinx
from dolfinx import fem, io
from dolfinx.fem import functionspace, Function, Constant
from dolfinx.fem.petsc import LinearProblem
import ufl
from ufl import inner, grad, TestFunction, TrialFunction

from geometry import BAT_TAGS, PCM_TAG, OUTER_BC_TAG
from materials import (
    K_BAT, RHO_BAT, CP_BAT, q_generation,
    K_PCM_SOLID, RHO_PCM_SOLID, CP_PCM_SOLID, L_PCM,
    T_AMBIENT, H_CONV,
    K_NOPCM, RHO_CP_NOPCM,
    liquid_fraction, rho_cp_pcm, k_pcm, rho_pcm, cp_base_pcm,
)


# ── Result container ──────────────────────────────────────────────────────────
class TimeSeries(NamedTuple):
    """
    All time-series quantities recorded during one simulation run.

    Energy quantities are per metre of axial cell depth [J/m].
    """
    times:        np.ndarray   # simulation time [s]
    T_bat_max:    np.ndarray   # max  temperature in any battery [K]
    T_bat_avg:    np.ndarray   # mean temperature over battery DOFs [K]
    T_pcm_avg:    np.ndarray   # cell-averaged mean PCM temperature [K]
    liq_fraction: np.ndarray   # volume-averaged liquid fraction [–]
    Q_latent:     np.ndarray   # latent   heat stored in PCM [J/m]
    Q_sensible:   np.ndarray   # sensible heat stored in PCM [J/m]
    Q_total_pcm:  np.ndarray   # Q_latent + Q_sensible        [J/m]
    E_generated:  np.ndarray   # cumulative battery energy     [J/m]


# ── Solver class ──────────────────────────────────────────────────────────────
class BatteryPCMSolver:
    """
    Backward-Euler FEM solver for the battery–PCM heat-conduction problem.
    """

    def __init__(
        self,
        msh,
        cell_tags,
        facet_tags,
        dt:         float = 30.0,
        t_end:      float = 1800.0,
        output_dir: Path  = Path("output"),
        use_pcm:    bool  = True,
        save_every: int   = 5,
    ) -> None:

        self.msh        = msh
        self.cell_tags  = cell_tags
        self.facet_tags = facet_tags
        self.dt_val     = dt
        self.t_end      = t_end
        self.output_dir = output_dir
        self.use_pcm    = use_pcm
        self.save_every = save_every

        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect battery cell indices
        self.bat_cells = np.concatenate(
            [cell_tags.find(t) for t in BAT_TAGS]
        ).astype(np.int32)

        # ── Define domains, mesh, and function spaces ─────────────────────────
        if not self.use_pcm:
            # WITHOUT PCM: Extract battery submesh
            self.active_mesh, self.entity_map, self.vertex_map, self.geom_map = dolfinx.mesh.create_submesh(
                msh, msh.topology.dim, self.bat_cells
            )
            # Find boundary facets of battery cells to apply natural convection
            boundary_facets = dolfinx.mesh.locate_entities_boundary(
                self.active_mesh, self.active_mesh.topology.dim - 1,
                lambda x: np.ones(x.shape[1], dtype=bool)
            )
            facet_indices = boundary_facets
            facet_values = np.ones_like(facet_indices, dtype=np.int32)
            sorted_indices = np.argsort(facet_indices)
            self.bat_facet_tags = dolfinx.mesh.meshtags(
                self.active_mesh, self.active_mesh.topology.dim - 1,
                facet_indices[sorted_indices], facet_values[sorted_indices]
            )
            
            self.V  = functionspace(self.active_mesh, ("Lagrange", 1))
            self.V0 = functionspace(self.active_mesh, ("DG",       0))
        else:
            # WITH PCM: Use the full composite mesh
            self.active_mesh = msh
            self.pcm_cells   = cell_tags.find(PCM_TAG).astype(np.int32)
            self.V  = functionspace(msh, ("Lagrange", 1))
            self.V0 = functionspace(msh, ("DG",       0))

        # Pre-build vectorised cell-to-dof mapping
        dm = self.V.dofmap
        self._cell_dofs = np.asarray(dm.list)   # (n_cells, 3) for CG1 triangles

        # Identify DOF subset for the battery region
        if self.use_pcm:
            self._bat_dofs = np.unique(
                self._cell_dofs[self.bat_cells].ravel()
            )
        else:
            self._bat_dofs = np.arange(self.V.dofmap.index_map.size_local, dtype=np.int32)

        # ── DG0 Material Property Fields ──────────────────────────────────────
        self.k_f    = Function(self.V0, name="ThermalConductivity")
        self.rho_cp = Function(self.V0, name="RhoCp")
        self.q_f    = Function(self.V0, name="HeatGeneration")
        self._init_properties()

        # ── Temperature Fields ────────────────────────────────────────────────
        self.T_n = Function(self.V, name="Temperature")
        self.T_h = Function(self.V, name="Temperature")
        self.T_n.x.array[:] = T_AMBIENT
        self.T_n.x.scatter_forward()

        # ── PDE constants and measures ────────────────────────────────────────
        dt_c   = Constant(self.active_mesh, float(dt))
        h_c    = Constant(self.active_mesh, H_CONV)
        Tinf_c = Constant(self.active_mesh, T_AMBIENT)

        dx_all = ufl.Measure("dx", domain=self.active_mesh)

        if self.use_pcm:
            self.ds_m = ufl.Measure("ds", domain=self.active_mesh, subdomain_data=self.facet_tags)
            self.dx_m = ufl.Measure("dx", domain=self.active_mesh, subdomain_data=self.cell_tags)
            ds_bc  = self.ds_m(OUTER_BC_TAG)
        else:
            ds_bc  = ufl.Measure("ds", domain=self.active_mesh, subdomain_data=self.bat_facet_tags)(1)

        # ── Variational form ──────────────────────────────────────────────────
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        a = (
            (self.rho_cp / dt_c) * u * v * dx_all
            + self.k_f * inner(grad(u), grad(v)) * dx_all
            + h_c * u * v * ds_bc
        )

        L = (
            (self.rho_cp / dt_c) * self.T_n * v * dx_all
            + self.q_f * v * dx_all
            + h_c * Tinf_c * v * ds_bc
        )

        self._problem = LinearProblem(
            a, L,
            bcs=[],
            u=self.T_h,
            petsc_options={
                "ksp_type":      "cg",
                "pc_type":       "hypre",
                "pc_hypre_type": "boomeramg",
                "ksp_rtol":      1e-8,
                "ksp_max_it":    1000,
            },
        )

        # ── Output file ───────────────────────────────────────────────────────
        fname = "temperature.xdmf" if use_pcm else "temperature_nopcm.xdmf"
        self._xdmf = io.XDMFFile(self.active_mesh.comm, str(output_dir / fname), "w")
        self._xdmf.write_mesh(self.active_mesh)

        # ── Integrals ─────────────────────────────────────────────────────────
        if self.use_pcm:
            one = Constant(self.active_mesh, 1.0)
            self._vol_pcm = fem.assemble_scalar(fem.form(one * self.dx_m(PCM_TAG)))
            self._vol_bat = sum(
                fem.assemble_scalar(fem.form(one * self.dx_m(t)))
                for t in BAT_TAGS
            )

            # Pre-compiled forms for energy integration
            self._fl_dg    = Function(self.V0, name="LiquidFraction")
            self._q_lat_dg = Function(self.V0, name="LatentHeatDensity")
            self._q_sen_dg = Function(self.V0, name="SensibleHeatDensity")

            self._fl_form    = fem.form(self._fl_dg    * self.dx_m(PCM_TAG))
            self._q_lat_form = fem.form(self._q_lat_dg * self.dx_m(PCM_TAG))
            self._q_sen_form = fem.form(self._q_sen_dg * self.dx_m(PCM_TAG))
        else:
            one = Constant(self.active_mesh, 1.0)
            self._vol_bat = fem.assemble_scalar(fem.form(one * dx_all))
            self._vol_pcm = 1.0   # placeholder to avoid divide-by-zero

    # ── Private helpers ────────────────────────────────────────────────────────

    def _init_properties(self) -> None:
        """Fill DG0 property fields with initial values."""
        k_a   = self.k_f.x.array
        rcp_a = self.rho_cp.x.array
        q_a   = self.q_f.x.array

        if self.use_pcm:
            # Full composite domain
            k_a[self.bat_cells]   = K_BAT
            rcp_a[self.bat_cells] = RHO_BAT * CP_BAT
            q_a[self.bat_cells]   = q_generation(0.0)

            k_a[self.pcm_cells]   = K_PCM_SOLID
            rcp_a[self.pcm_cells] = RHO_PCM_SOLID * CP_PCM_SOLID
            q_a[self.pcm_cells]   = 0.0
        else:
            # Battery-only domain
            k_a[:]   = K_BAT
            rcp_a[:] = RHO_BAT * CP_BAT
            q_a[:]   = q_generation(0.0)

        for f in (self.k_f, self.rho_cp, self.q_f):
            f.x.scatter_forward()

    def _cell_avg_T(self) -> np.ndarray:
        """Compute average temperature of each cell."""
        return self.T_n.x.array[self._cell_dofs].mean(axis=1)

    def _update_pcm_props(self, T_cell: np.ndarray) -> None:
        """Update temperature-dependent PCM properties (WITH PCM mode only)."""
        if not self.use_pcm:
            return
        T_pcm = T_cell[self.pcm_cells]
        self.k_f.x.array[self.pcm_cells]    = k_pcm(T_pcm)
        self.rho_cp.x.array[self.pcm_cells] = rho_cp_pcm(T_pcm)

        self.k_f.x.scatter_forward()
        self.rho_cp.x.scatter_forward()

    # ── Time loop ──────────────────────────────────────────────────────────────

    def run(self) -> TimeSeries:
        n_steps = int(round(self.t_end / self.dt_val))

        times       = np.zeros(n_steps + 1)
        T_bat_max   = np.zeros(n_steps + 1)
        T_bat_avg   = np.zeros(n_steps + 1)
        T_pcm_avg   = np.zeros(n_steps + 1)
        liq_frac    = np.zeros(n_steps + 1)
        Q_latent    = np.zeros(n_steps + 1)
        Q_sensible  = np.zeros(n_steps + 1)
        E_generated = np.zeros(n_steps + 1)

        T_cell = self._cell_avg_T()
        self._record(0, 0.0, T_cell,
                     times, T_bat_max, T_bat_avg, T_pcm_avg,
                     liq_frac, Q_latent, Q_sensible, E_generated)
        self._xdmf.write_function(self.T_n, 0.0)

        t = 0.0
        label = "PCM" if self.use_pcm else "NOPCM"

        for step in range(1, n_steps + 1):
            t += self.dt_val

            T_cell = self._cell_avg_T()
            self._update_pcm_props(T_cell)

            # Update time-dependent heat source
            if self.use_pcm:
                self.q_f.x.array[self.bat_cells] = q_generation(t)
            else:
                self.q_f.x.array[:] = q_generation(t)
            self.q_f.x.scatter_forward()

            # Solve
            self._problem.solve()

            # Advance
            self.T_n.x.array[:] = self.T_h.x.array
            self.T_n.x.scatter_forward()

            T_cell = self._cell_avg_T()
            self._record(step, t, T_cell,
                         times, T_bat_max, T_bat_avg, T_pcm_avg,
                         liq_frac, Q_latent, Q_sensible, E_generated)

            if step % self.save_every == 0:
                self._xdmf.write_function(self.T_n, t)

            if step % 10 == 0 or step == 1:
                print(
                    f"  [{label}] t = {t:6.0f} s | "
                    f"T_bat_max = {T_bat_max[step] - 273.15:5.1f} °C | "
                    f"f_l = {liq_frac[step]:.4f}"
                )

        self._xdmf.close()
        print(f"\n[solver] Run complete ({label}). "
              f"XDMF saved → {self.output_dir}")

        return TimeSeries(
            times        = times,
            T_bat_max    = T_bat_max,
            T_bat_avg    = T_bat_avg,
            T_pcm_avg    = T_pcm_avg,
            liq_fraction = liq_frac,
            Q_latent     = Q_latent,
            Q_sensible   = Q_sensible,
            Q_total_pcm  = Q_latent + Q_sensible,
            E_generated  = E_generated,
        )

    def _record(
        self, step: int, t: float, T_cell: np.ndarray,
        times, T_bat_max, T_bat_avg, T_pcm_avg,
        liq_frac, Q_latent, Q_sensible, E_generated,
    ) -> None:
        T_a = self.T_n.x.array
        times[step] = t

        # Battery temperatures
        T_bat_vals      = T_a[self._bat_dofs]
        T_bat_max[step] = float(T_bat_vals.max())
        T_bat_avg[step] = float(T_bat_vals.mean())

        # PCM integrals
        if self.use_pcm:
            T_pcm_vals      = T_cell[self.pcm_cells]
            T_pcm_avg[step] = float(T_pcm_vals.mean())

            fl_vals = liquid_fraction(T_pcm_vals)
            self._fl_dg.x.array[self.pcm_cells] = fl_vals
            self._fl_dg.x.scatter_forward()
            
            rho_vals = rho_pcm(T_pcm_vals)
            cp_vals = cp_base_pcm(T_pcm_vals)
            
            self._q_lat_dg.x.array[self.pcm_cells] = rho_vals * L_PCM * fl_vals
            self._q_lat_dg.x.scatter_forward()
            
            self._q_sen_dg.x.array[self.pcm_cells] = rho_vals * cp_vals * (T_pcm_vals - T_AMBIENT)
            self._q_sen_dg.x.scatter_forward()

            fl_integral      = fem.assemble_scalar(self._fl_form)
            liq_frac[step]   = fl_integral / self._vol_pcm
            Q_latent[step]   = fem.assemble_scalar(self._q_lat_form)
            Q_sensible[step] = fem.assemble_scalar(self._q_sen_form)
        else:
            T_pcm_avg[step]  = T_AMBIENT
            liq_frac[step]   = 0.0
            Q_latent[step]   = 0.0
            Q_sensible[step] = 0.0

        # Cumulative energy integration
        if step == 0:
            E_generated[step] = 0.0
        else:
            E_generated[step] = E_generated[step - 1] + q_generation(t) * self._vol_bat * self.dt_val
