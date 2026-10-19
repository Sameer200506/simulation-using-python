"""
materials.py
------------
Thermophysical property definitions for:
  • 26650 Li-ion battery (homogenised, isotropic composite electrode stack)
  • Phase change material  – RT-42 paraffin wax (Rubitherm GmbH)
  • Air                    – used for the no-PCM reference run

PCM phase-change is captured by the *apparent heat capacity* method:

    ρ c_p_eff(T)  =  ρ c_p_base  +  ρ L  ∂f_l/∂T(T)

where f_l(T) is a piecewise-linear liquid fraction smoothed over the mushy
zone [T_solidus, T_liquidus].  The peak value of c_p_eff is L/(T_l − T_s).

All values are in SI units.  All array functions accept scalar or ndarray input
and return the same type, enabling both cell-level and vectorised evaluation.
"""

from __future__ import annotations
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  26650 Li-ion battery (homogenised composite)
# ══════════════════════════════════════════════════════════════════════════════
K_BAT   = 1.5        # W/(m·K)    effective through-plane conductivity
RHO_BAT = 2500.0     # kg/m³      average electrode-stack density
CP_BAT  = 1000.0     # J/(kg·K)   average specific heat

# ── 2C discharge volumetric heat generation data (W/m³) vs time (s) ───────────
QGEN_TIMES = np.array([
    0.0, 94.7, 189.5, 284.2, 378.9, 473.7, 568.4, 663.2, 757.9, 852.6,
    947.4, 1042.1, 1136.8, 1231.6, 1326.3, 1421.1, 1515.8, 1610.5, 1705.3, 1800.0
])
QGEN_VALUES = np.array([
    49622.73, 51712.11, 53801.49, 55890.86, 57980.24, 60069.62, 62159.00, 64248.38,
    66337.75, 68427.13, 70516.51, 72605.89, 74695.27, 76784.65, 78874.02, 80963.40,
    83052.78, 85142.16, 87231.54, 89320.91
])

def q_generation(t: float) -> float:
    """Get volumetric heat generation [W/m³] at time t via linear interpolation."""
    # Extrapolate constant
    return float(np.interp(t, QGEN_TIMES, QGEN_VALUES))

# ══════════════════════════════════════════════════════════════════════════════
#  PCM properties
# ══════════════════════════════════════════════════════════════════════════════
K_PCM_SOLID  = 3.0      # W/(m·K)
K_PCM_LIQUID = 2.5      # W/(m·K)
RHO_PCM_SOLID  = 850.0  # kg/m³
RHO_PCM_LIQUID = 780.0  # kg/m³
CP_PCM_SOLID   = 2000.0 # J/(kg·K)
CP_PCM_LIQUID  = 2400.0 # J/(kg·K)
L_PCM          = 140_000 # J/kg    latent heat of fusion

T_MELT_CENTER  = 328.15 # K  (55 °C)  phase change temperature
T_INTERVAL     = 5.0    # K  transition interval
T_SOLIDUS      = T_MELT_CENTER - T_INTERVAL / 2.0  # 325.65 K (52.5 °C)
T_LIQUIDUS     = T_MELT_CENTER + T_INTERVAL / 2.0  # 330.65 K (57.5 °C)

# ══════════════════════════════════════════════════════════════════════════════
#  Boundary / ambient conditions
# ══════════════════════════════════════════════════════════════════════════════
T_AMBIENT = 298.15   # K  (25 °C)  initial temperature and ambient temperature
H_CONV    = 5.0      # W/(m²·K)   natural-convection heat-transfer coefficient

# ══════════════════════════════════════════════════════════════════════════════
#  Air properties (no-PCM reference run)
# ══════════════════════════════════════════════════════════════════════════════
K_NOPCM      = 0.025          # W/(m·K) (air thermal conductivity)
RHO_CP_NOPCM = 1.2 * 1005.0   # J/(m³·K) (air density * specific heat)


# ── Temperature-dependent property functions ──────────────────────────────────

def liquid_fraction(T: "np.ndarray | float") -> "np.ndarray | float":
    """
    Liquid fraction f_l(T) – linear ramp in the mushy zone.

      f_l = 0                            for T ≤ T_SOLIDUS
      f_l = (T − T_s) / (T_l − T_s)    for T_SOLIDUS < T < T_LIQUIDUS
      f_l = 1                            for T ≥ T_LIQUIDUS

    Parameters
    ----------
    T : temperature [K], scalar or ndarray.

    Returns
    -------
    f_l : same shape as T, clipped to [0, 1].
    """
    scalar = np.isscalar(T)
    T_arr  = np.atleast_1d(np.asarray(T, dtype=float))
    fl     = np.clip(
        (T_arr - T_SOLIDUS) / (T_LIQUIDUS - T_SOLIDUS),
        0.0, 1.0
    )
    return float(fl[0]) if scalar else fl


def d_fl_dT(T: "np.ndarray | float") -> "np.ndarray | float":
    """
    Derivative ∂f_l/∂T [K⁻¹] – forms the rectangular peak in c_p_eff.

    Equal to 1/(T_l − T_s) inside the mushy zone, 0 elsewhere.
    """
    scalar = np.isscalar(T)
    T_arr  = np.atleast_1d(np.asarray(T, dtype=float))
    inv_dT = 1.0 / (T_LIQUIDUS - T_SOLIDUS)
    deriv  = np.where(
        (T_arr > T_SOLIDUS) & (T_arr < T_LIQUIDUS),
        inv_dT, 0.0
    )
    return float(deriv[0]) if scalar else deriv


def rho_pcm(T: "np.ndarray | float") -> "np.ndarray | float":
    """Temperature-dependent density of PCM [kg/m³]."""
    fl = liquid_fraction(T)
    return RHO_PCM_SOLID + fl * (RHO_PCM_LIQUID - RHO_PCM_SOLID)


def cp_base_pcm(T: "np.ndarray | float") -> "np.ndarray | float":
    """Sensible specific heat capacity of PCM [J/(kg·K)]."""
    fl = liquid_fraction(T)
    return CP_PCM_SOLID + fl * (CP_PCM_LIQUID - CP_PCM_SOLID)


def cp_eff_pcm(T: "np.ndarray | float") -> "np.ndarray | float":
    """
    Effective (apparent) heat capacity of PCM [J/(kg·K)].

    c_p_eff = c_p_base + L · ∂f_l/∂T(T)
    """
    return cp_base_pcm(T) + L_PCM * d_fl_dT(T)


def k_pcm(T: "np.ndarray | float") -> "np.ndarray | float":
    """
    Temperature-dependent thermal conductivity of PCM [W/(m·K)].

    Linear interpolation between solid and liquid conductivities via f_l.
    """
    fl = liquid_fraction(T)
    return K_PCM_SOLID + fl * (K_PCM_LIQUID - K_PCM_SOLID)


def rho_cp_pcm(T: "np.ndarray | float") -> "np.ndarray | float":
    """ρ × c_p_eff for PCM [J/(m³·K)]."""
    return rho_pcm(T) * cp_eff_pcm(T)


# ── Self-check ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    T_range = np.linspace(315, 345, 500)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(T_range - 273.15, cp_eff_pcm(T_range) / 1e3)
    axes[0].set_xlabel("T [°C]");  axes[0].set_ylabel("c_p_eff [kJ/(kg·K)]")
    axes[0].set_title("Apparent heat capacity")

    axes[1].plot(T_range - 273.15, k_pcm(T_range))
    axes[1].set_xlabel("T [°C]");  axes[1].set_ylabel("k [W/(m·K)]")
    axes[1].set_title("PCM conductivity")

    axes[2].plot(T_range - 273.15, liquid_fraction(T_range) * 100)
    axes[2].set_xlabel("T [°C]");  axes[2].set_ylabel("f_l [%]")
    axes[2].set_title("Liquid fraction")

    plt.tight_layout()
    plt.savefig("materials_check.png", dpi=120)
    print("Saved materials_check.png")
