"""
core/solver.py -- LOD (Locally One-Dimensional) solver with Numba JIT.

Same Yanenko operator-splitting physics as before, but the Thomas algorithm
inner loops are compiled to machine code with Numba @njit(parallel=True).

Performance
-----------
The Thomas sweep for a 50x50x50 grid:
  - x-sweep: 50 equations, 50*50=2500 independent columns
  - Numba prange: all 2500 solved in parallel across CPU cores
  - Expected speedup over pure Python: 15-40x

First run: ~3s compilation (cached to disk after that).
All subsequent runs: native machine-code speed.

GPU note
--------
No CUDA GPU detected on this system. If a CUDA GPU is added later:
  - Replace @njit(parallel=True) with @cuda.jit for the Thomas kernel
  - Or install cupy and use cupyx tridiagonal solver
  Currently all acceleration is CPU-side (all cores via OpenMP through Numba).
"""
import time
import numpy as np
import config
from .grid import Grid

# ── Numba import with graceful fallback ───────────────────────────────────────
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    # ponytail: if numba not installed, fall back to pure numpy.
    # The solver still works, just slower.
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(fn): return fn
        return decorator
    prange = range


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    return np.where(denom > 0, 2.0 * a * b / denom, 0.0)


# ── Core Thomas algorithm — Numba-compiled, parallel over columns ─────────────

@njit(parallel=True, fastmath=True, cache=True)
def _thomas_batch(a: np.ndarray, b_in: np.ndarray,
                  c: np.ndarray, D_in: np.ndarray) -> np.ndarray:
    """
    Solve M independent tridiagonal systems in parallel.

    a[i,m]*x[i-1] + b[i,m]*x[i] + c[i,m]*x[i+1] = D[i,m]

    Each column m is independent — solved in parallel across CPU cores.
    Thread-local copies of b and D avoid race conditions.

    Parameters
    ----------
    a, b_in, c, D_in : shape (N_ax, M)

    Returns
    -------
    X : shape (N_ax, M)
    """
    N, M = D_in.shape
    X    = np.empty_like(D_in)

    for m in prange(M):                    # parallel over columns
        # Thread-local scratch (avoids writing to shared arrays)
        b = np.empty(N)
        D = np.empty(N)
        for i in range(N):
            b[i] = b_in[i, m]
            D[i] = D_in[i, m]

        # Forward elimination
        for i in range(1, N):
            w     = a[i, m] / b[i - 1]
            b[i] -= w * c[i - 1, m]
            D[i] -= w * D[i - 1]

        # Back substitution
        X[N - 1, m] = D[N - 1] / b[N - 1]
        for i in range(N - 2, -1, -1):
            X[i, m] = (D[i] - c[i, m] * X[i + 1, m]) / b[i]

    return X


class Solver:
    """
    LOD/Yanenko solver with Numba-parallel Thomas algorithm.
    Unconditionally stable for any dt.
    """

    def __init__(
        self,
        grid: Grid,
        dt: float       = config.DT,
        t_end: float    = config.T_END,
        t_amb: float    = config.T_AMB,
        h_conv: float   = config.H_CONV,
        save_every: int = config.SAVE_EVERY,
    ):
        self.grid       = grid
        self.dt         = dt
        self.t_end      = t_end
        self.t_amb      = t_amb
        self.h_conv     = h_conv
        self.save_every = save_every

        self.n_steps = int(round(t_end / dt))
        self.time    = 0.0
        self.step    = 0

        self.history_t:      list[float] = []
        self.history_T_max:  list[float] = []
        self.history_T_mean: list[float] = []
        self.snapshots:      list[tuple] = []

        g = grid
        self._kx  = _harmonic_mean(g.k_r[:-1, :, :], g.k_r[1:, :, :])
        self._ky  = _harmonic_mean(g.k_r[:, :-1, :], g.k_r[:, 1:, :])
        self._kz  = _harmonic_mean(g.k_z[:, :, :-1], g.k_z[:, :, 1:])
        self._rCp = g.rho * g.Cp

        accel = "Numba @njit parallel" if _HAS_NUMBA else "NumPy (install numba for speedup)"
        print(f"[Solver] LOD -- dt={dt}s, {self.n_steps} steps, "
              f"grid {g.nx}x{g.ny}x{g.nz} ({g.n_cells:,} cells)")
        print(f"[Solver] Acceleration: {accel}")
        self._report_explicit_limit()

        if _HAS_NUMBA:
            # Warm up JIT compilation on a tiny dummy problem
            print("[Solver] Compiling Numba Thomas kernel (first run only) ...")
            t0 = time.perf_counter()
            _dummy = np.ones((3, 4), dtype=np.float64)
            _thomas_batch(_dummy, _dummy * 3, _dummy, _dummy)
            print(f"[Solver] Numba compiled in {time.perf_counter()-t0:.1f}s")

    def _report_explicit_limit(self) -> None:
        g = self.grid
        alpha_max = (np.maximum(g.k_r, g.k_z) / self._rCp).max()
        dt_expl   = 1.0 / (2.0 * alpha_max * (1/g.dx**2 + 1/g.dy**2 + 1/g.dz**2))
        print(f"[Solver] Explicit dt limit: {dt_expl:.5f}s  |  "
              f"LOD dt: {self.dt:.2f}s  (implicit factor: {self.dt/dt_expl:.0f}x)")

    # ── LOD sweep ─────────────────────────────────────────────────────────────

    def _sweep(self, T: np.ndarray, axis: int) -> np.ndarray:
        """
        Implicit 1-D sweep in direction `axis`.
        No explicit cross-direction terms -- unconditionally stable.
        """
        g     = self.grid
        N_ax  = g.shape[axis]
        dx    = (g.dx, g.dy, g.dz)[axis]
        kface = (self._kx, self._ky, self._kz)[axis]

        other    = [i for i in range(3) if i != axis]
        perm     = [axis] + other
        inv_perm = list(np.argsort(perm))

        T_p    = np.ascontiguousarray(T.transpose(perm))
        rCp_p  = np.ascontiguousarray(self._rCp.transpose(perm))
        Q_p    = np.ascontiguousarray(g.Q.transpose(perm))
        kf_p   = np.ascontiguousarray(kface.transpose(perm))

        shape_p = T_p.shape
        M = shape_p[1] * shape_p[2]

        T_f   = T_p.reshape(N_ax, M)
        rCp_f = rCp_p.reshape(N_ax, M)
        Q_f   = Q_p.reshape(N_ax, M)
        kf_f  = kf_p.reshape(N_ax - 1, M)

        inv_d2 = 1.0 / dx ** 2
        bc     = self.h_conv / dx

        a = np.zeros((N_ax, M), dtype=np.float64)
        b = (rCp_f / self.dt).copy()
        c = np.zeros((N_ax, M), dtype=np.float64)

        a[1:,  :] = -kf_f * inv_d2
        c[:-1, :] = -kf_f * inv_d2
        b[1:,  :] +=  kf_f * inv_d2
        b[:-1, :] +=  kf_f * inv_d2

        b[0,  :] += bc
        b[-1, :] += bc

        D = rCp_f / self.dt * T_f + Q_f / 3.0
        D[0,  :] += bc * self.t_amb
        D[-1, :] += bc * self.t_amb

        # Use Numba-compiled Thomas if available, otherwise pure numpy fallback
        if _HAS_NUMBA:
            X = _thomas_batch(a, b, c, D)
        else:
            X = _thomas_numpy(a, b, c, D, N_ax)

        return X.reshape(shape_p).transpose(inv_perm)

    def _step(self) -> None:
        """One LOD time step: 3 direction sweeps."""
        T = self.grid.T
        T = self._sweep(T, axis=0)
        T = self._sweep(T, axis=1)
        T = self._sweep(T, axis=2)
        self.grid.T = T

    # ── Run loop ──────────────────────────────────────────────────────────────

    def run(
        self,
        callback=None,
        pack=None,          # optional Pack object for NTGK coupling
        verbose: bool = True,
    ) -> None:
        """
        Integrate from t=0 to t=t_end.

        Parameters
        ----------
        pack     : Pack instance — if provided, Q is updated from NTGK each step
        callback : callable(solver, step) called every save_every steps
        """
        t_wall_start = time.perf_counter()
        batt_mask = self.grid.material_id == 1

        for step in range(self.n_steps):
            # Update heat source from NTGK model (if pack provided)
            if pack is not None:
                pack.update_Q()

            self._step()

            # Advance electrochemical state
            if pack is not None:
                pack.advance(self.dt)

            self.time = (step + 1) * self.dt
            self.step = step + 1

            T_batt = self.grid.T[batt_mask]
            self.history_t.append(self.time)
            self.history_T_max.append(float(T_batt.max())   if T_batt.size else float('nan'))
            self.history_T_mean.append(float(T_batt.mean()) if T_batt.size else float('nan'))

            if step % self.save_every == 0:
                self.snapshots.append((self.time, self.grid.T.copy()))
                if verbose:
                    elapsed = time.perf_counter() - t_wall_start
                    pct     = 100.0 * (step + 1) / self.n_steps
                    T_max   = self.history_T_max[-1]
                    extra   = ""
                    if pack is not None:
                        s = pack.stats()
                        extra = (f"  DOD={s['DOD']:.3f}"
                                 f"  V={s['V_terminal_V']:.2f}V/cell"
                                 f"  hot=cell{s['hottest_cell']}")
                    print(
                        f"  t={self.time:8.1f}s  ({pct:5.1f}%)  "
                        f"T_max={T_max - 273.15:6.2f}C  "
                        f"wall={elapsed:.1f}s{extra}"
                    )
                if callback is not None:
                    callback(self, step)

        t_wall = time.perf_counter() - t_wall_start
        print(f"\n[Solver] Done -- {self.n_steps} steps in {t_wall:.2f}s wall-time")
        print(f"         Final T_max = {self.history_T_max[-1] - 273.15:.2f}C")


def _thomas_numpy(a, b_in, c, D_in, N_ax):
    """Pure-numpy fallback when Numba is unavailable."""
    b_ = b_in.copy()
    D_ = D_in.copy()
    c_ = c.copy()
    for i in range(1, N_ax):
        w      = a[i] / b_[i - 1]
        b_[i] -= w * c_[i - 1]
        D_[i] -= w * D_[i - 1]
    X = D_.copy()
    X[-1] /= b_[-1]
    for i in range(N_ax - 2, -1, -1):
        X[i] = (D_[i] - c_[i] * X[i + 1]) / b_[i]
    return X
