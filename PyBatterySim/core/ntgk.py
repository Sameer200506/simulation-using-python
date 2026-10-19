"""
core/ntgk.py -- NTGK (Newman-Tiedemann-Gu-Kim) electrochemical heat model.

The NTGK model is a semi-empirical battery model widely used in CFD codes
(ANSYS Fluent, Star-CCM+) because it couples electrochemistry to a thermal
solver with only a few polynomial coefficients.

Physics
-------
State variable:   DOD(t)  -- Depth of Discharge  [0 = full, 1 = empty]
                  dDOD/dt = I / (C_nom * 3600)

Polynomial parameters (fitted from discharge curve data):
    U_OCV(DOD) = sum(a_k * DOD^k, k=0..n)   open-circuit voltage [V/cell]
    R_int(DOD) = internal resistance [Ohm/cell]  (can be DOD-dependent)

Heat generation per cell [W]:
    Q_joule    = I^2 * R_int(DOD)            -- irreversible Joule heating
    Q_entropy  = I * T_cell * dU/dT          -- reversible entropic heat
                 (dU/dT ~ -3e-4 V/K for most LiB, sign depends on direction)
    Q_total    = Q_joule + |Q_entropy|

Volumetric heat [W/m^3]:
    q = Q_total / V_cell

Usage
-----
    ntgk = NTGKModel("LIB.xlsx")          # fits polynomials from data
    ntgk.reset()                           # reset DOD=0 for a new simulation
    q_vol = ntgk.heat_per_m3(I, T_cell)   # call at each time step
    ntgk.advance(dt)                       # advance DOD by one time step
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.polynomial import polynomial as P


# Typical LiB entropic coefficient [V/K] -- negative means heat absorbed on charge
_ENTROPY_COEFF_V_PER_K: float = -3.0e-4


class NTGKModel:
    """
    NTGK electrochemical model for a single 26650 cell.

    Parameters
    ----------
    xlsx_path  : path to LIB.xlsx; Sheet2 should be the discharge curve
    C_nom      : nominal capacity [Ah]
    R_int      : internal resistance [Ohm/cell] -- if None, estimated from data
    poly_order : order of the U_OCV(DOD) polynomial fit
    """

    def __init__(
        self,
        xlsx_path: str | None = None,
        C_nom: float       = 4.5,      # Ah
        R_int: float | None = None,     # Ohm; None -> auto-estimate or use default
        poly_order: int    = 5,
        n_cells_series: int = 4,       # cells in series (4S pack)
        sheet_discharge: str = "Sheet2",
    ):
        self.C_nom           = C_nom
        self.poly_order      = poly_order
        self.n_series        = n_cells_series
        self._dod            = 0.0      # current DOD, updated by advance()

        if xlsx_path is not None:
            # Load and fit from Excel discharge data
            self._fit_from_excel(xlsx_path, sheet_discharge, R_int)
            print(f"[NTGK] Fitted U_OCV polynomial (order {poly_order}) from {xlsx_path}")
        else:
            # Fallback/Default pre-fitted parameters (from LIB.xlsx Sheet2)
            # highest power first: [dod^5, dod^4, dod^3, dod^2, dod^1, constant]
            self._u_coeffs = np.array([
                -6.16393174, 17.27962311, -17.79881149, 
                7.83413887, -1.92050681, 3.94740274
            ])
            self._du_coeffs = np.polyder(np.poly1d(self._u_coeffs)).coefficients
            self.R_int = R_int if R_int is not None else 0.017  # 17.0 mOhm
            self._I_estimated = 10.72  # A (from 1511s test duration)
            self._dod_max = 1.0
            print("[NTGK] Using default pre-fitted U_OCV coefficients (No Excel required)")

        print(f"       R_int = {self.R_int*1000:.1f} mOhm/cell   "
              f"I_discharge = {self._I_estimated:.2f} A  "
              f"C_nom = {C_nom} Ah")

    # ── Data fitting ──────────────────────────────────────────────────────────

    def _fit_from_excel(
        self, path: str, sheet: str, R_int_override: float | None
    ) -> None:
        df = pd.ExcelFile(path).parse(sheet, header=None)
        # Column 1 = pack voltage, Column 2 = temperature (C)
        V_pack = df.iloc[:, 1].values.astype(float)   # V (4S pack)
        T_data = df.iloc[:, 2].values.astype(float)   # deg C

        n = len(V_pack)
        # DOD goes from 0 to 1 uniformly (constant-current assumption)
        dod = np.linspace(0.0, 1.0, n)

        # Per-cell voltage
        V_cell = V_pack / self.n_series

        # Estimate discharge time and current
        # Capacity drawn: C_nom Ah at some C-rate
        # t_discharge = n seconds (1 row = 1 second assumption)
        t_discharge = float(n)                    # seconds
        self._I_estimated = self.C_nom * 3600.0 / t_discharge   # A

        # Estimate R_int from energy balance:
        # Q_total approx m*Cp * dT/dt  (ignoring convection: lower bound)
        # Then R_int = Q_total / I^2
        rho, Cp = 2500.0, 1000.0
        r, h    = 0.013, 0.065
        V_batt  = np.pi * r**2 * h
        m_batt  = rho * V_batt
        dT_total = T_data[-1] - T_data[0]   # K rise
        Q_avg    = m_batt * Cp * dT_total / t_discharge   # W (lower bound)
        R_est    = Q_avg / (self._I_estimated ** 2)

        self.R_int = R_int_override if R_int_override is not None else max(R_est, 0.010)

        # Correct V_cell to approximate U_OCV:
        # U_OCV = V_terminal + I * R_int  (discharge: terminal is lower than OCV)
        U_ocv = V_cell + self._I_estimated * self.R_int

        # Fit polynomial: U_OCV(DOD)
        # Using numpy polynomial (coefficient order: [a0, a1, a2, ...] for a0 + a1*x + ...)
        self._u_coeffs = np.polyfit(dod, U_ocv, self.poly_order)
        # np.polyfit gives highest-degree first; we use np.polyval which also expects that

        # dU/dDOD polynomial (derivative of U_OCV)
        self._du_coeffs = np.polyder(np.poly1d(self._u_coeffs)).coefficients

        # Store DOD range for clamping
        self._dod_max = 1.0

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, dod: float = 0.0) -> None:
        """Reset DOD to start of discharge (or any initial DOD)."""
        self._dod = float(dod)

    @property
    def dod(self) -> float:
        return self._dod

    def U_ocv(self, dod: float | np.ndarray | None = None) -> float | np.ndarray:
        """Open-circuit voltage [V/cell] at given DOD."""
        d = self._dod if dod is None else dod
        return float(np.polyval(self._u_coeffs, d))

    def V_terminal(self, I: float, dod: float | None = None) -> float:
        """Terminal voltage [V/cell] during discharge at current I [A]."""
        return self.U_ocv(dod) - I * self.R_int

    def heat_per_cell(self, I: float, T_cell_K: float,
                      dod: float | None = None) -> float:
        """
        Total heat generated by one cell [W].

        Q_joule   = I^2 * R_int          (irreversible)
        Q_entropy = I * T * dU/dT        (reversible entropic heat)
                    Using dU/dT = _ENTROPY_COEFF_V_PER_K (constant approx)
        """
        d = self._dod if dod is None else dod
        Q_joule   = I ** 2 * self.R_int
        Q_entropy = abs(I * T_cell_K * _ENTROPY_COEFF_V_PER_K)
        return Q_joule + Q_entropy

    def heat_per_m3(self, I: float, T_cell_K: float,
                    dod: float | None = None) -> float:
        """Volumetric heat generation [W/m^3] for a 26650 cell."""
        r, h    = 0.013, 0.065
        V_batt  = np.pi * r**2 * h
        return self.heat_per_cell(I, T_cell_K, dod) / V_batt

    def advance(self, I: float, dt: float) -> None:
        """Advance DOD by one time step. Call after each solver step."""
        self._dod = min(
            self._dod + I * dt / (self.C_nom * 3600.0),
            self._dod_max
        )

    def summary(self) -> dict:
        return {
            "DOD":           self._dod,
            "U_OCV_V":       self.U_ocv(),
            "R_int_mOhm":    self.R_int * 1000,
            "C_nom_Ah":      self.C_nom,
            "I_est_A":       self._I_estimated,
        }
