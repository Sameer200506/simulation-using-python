"""
postprocess.py
--------------
Post-processing, visualisation and data export for the PCM-battery simulation.

Produces five publication-quality figures:
  1. Battery temperature vs time     – PCM and no-PCM comparison
  2. PCM average temperature vs time – with mushy-zone band
  3. Maximum battery temperature     – PCM vs no-PCM
  4. Heat absorbed by PCM            – total, sensible, latent + battery energy
  5. PCM liquid fraction vs time

All figures are saved as PNG in output_dir.
All scalars are exported to results.csv.
A formatted summary is printed to stdout.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")    # non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from solver import TimeSeries
from materials import T_SOLIDUS, T_LIQUIDUS


# ── Global Matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.fontsize":   10,
    "legend.framealpha": 0.85,
    "lines.linewidth":   2.0,
    "grid.alpha":        0.30,
    "grid.linestyle":    "--",
})

# Colour palette
C_PCM_MAX    = "#d62728"   # red
C_PCM_AVG    = "#ff7f0e"   # orange
C_NOP_MAX    = "#1f77b4"   # blue
C_NOP_AVG    = "#aec7e8"   # light blue
C_PCM_T      = "#2ca02c"   # green
C_TOTAL      = "#9467bd"   # purple
C_SENSIBLE   = "#e377c2"   # pink
C_LATENT     = "#17becf"   # teal
C_ENERGY     = "#7f7f7f"   # grey
C_LIQ        = "#17becf"   # teal


class PostProcessor:
    """
    Generate plots, CSV export and summary for a pair of simulation runs.

    Parameters
    ----------
    ts_pcm   : TimeSeries from the PCM run.
    ts_nopcm : TimeSeries from the no-PCM (air) reference run.
    output_dir : directory to write all output files.
    """

    def __init__(
        self,
        ts_pcm:    TimeSeries,
        ts_nopcm:  TimeSeries,
        output_dir: Path,
    ) -> None:
        self.pcm    = ts_pcm
        self.nopcm  = ts_nopcm
        self.outdir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert simulation time to minutes for cleaner x-axis labels
        self.t_min_pcm = ts_pcm.times   / 60.0
        self.t_min_nop = ts_nopcm.times / 60.0

        # Interpolate no-PCM results onto the PCM time grid
        self._nop_max_interp = np.interp(
            ts_pcm.times, ts_nopcm.times, ts_nopcm.T_bat_max
        )
        self._nop_avg_interp = np.interp(
            ts_pcm.times, ts_nopcm.times, ts_nopcm.T_bat_avg
        )
        # Temperature reduction: positive means PCM keeps batteries cooler
        self.delta_T = self._nop_max_interp - ts_pcm.T_bat_max

    # ── Figure 1: Battery temperature vs time ─────────────────────────────────

    def _fig1_battery_temperature(self) -> None:
        """Battery max and average temperature vs time (PCM vs no-PCM)."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.t_min_pcm, self.pcm.T_bat_max   - 273.15,
                color=C_PCM_MAX,  label="T_bat_max  (with PCM)")
        ax.plot(self.t_min_pcm, self.pcm.T_bat_avg   - 273.15,
                color=C_PCM_AVG,  label="T_bat_avg  (with PCM)", linestyle="--")
        ax.plot(self.t_min_nop, self.nopcm.T_bat_max - 273.15,
                color=C_NOP_MAX,  label="T_bat_max  (no PCM)")
        ax.plot(self.t_min_nop, self.nopcm.T_bat_avg - 273.15,
                color=C_NOP_AVG,  label="T_bat_avg  (no PCM)", linestyle="--")

        # Mushy-zone band
        ax.axhspan(T_SOLIDUS - 273.15, T_LIQUIDUS - 273.15,
                   color="gold", alpha=0.18, label="PCM mushy zone")

        ax.set_xlabel("Time  [min]")
        ax.set_ylabel("Temperature  [°C]")
        ax.set_title("Battery Temperature vs Time")
        ax.legend(loc="upper left")
        ax.grid(True)
        fig.tight_layout()
        self._save(fig, "fig1_battery_temperature.png")

    # ── Figure 2: PCM temperature vs time ────────────────────────────────────

    def _fig2_pcm_temperature(self) -> None:
        """PCM cell-averaged temperature vs time with mushy-zone band."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.t_min_pcm, self.pcm.T_pcm_avg - 273.15,
                color=C_PCM_T, label="PCM avg. temperature")

        ax.axhspan(T_SOLIDUS - 273.15, T_LIQUIDUS - 273.15,
                   color="gold", alpha=0.25, label="Mushy zone  (40–44 °C)")
        ax.axhline(T_SOLIDUS  - 273.15, color="goldenrod",
                   linestyle=":", linewidth=1.5)
        ax.axhline(T_LIQUIDUS - 273.15, color="darkorange",
                   linestyle=":", linewidth=1.5)

        ax.set_xlabel("Time  [min]")
        ax.set_ylabel("PCM Temperature  [°C]")
        ax.set_title("PCM Average Temperature vs Time")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self._save(fig, "fig2_pcm_temperature.png")

    # ── Figure 3: Max battery temperature ────────────────────────────────────

    def _fig3_max_battery_temperature(self) -> None:
        """Maximum battery temperature: PCM vs no-PCM and ΔT on twin axis."""
        fig, ax1 = plt.subplots(figsize=(8, 5))

        ax1.plot(self.t_min_pcm, self.pcm.T_bat_max - 273.15,
                 color=C_PCM_MAX, label="With PCM")
        ax1.plot(self.t_min_nop, self.nopcm.T_bat_max - 273.15,
                 color=C_NOP_MAX, label="No PCM (air)")
        ax1.set_xlabel("Time  [min]")
        ax1.set_ylabel("Max Battery Temperature  [°C]")
        ax1.set_title("Maximum Battery Temperature vs Time")
        ax1.grid(True)

        # Twin axis for ΔT (temperature reduction by PCM)
        ax2 = ax1.twinx()
        ax2.fill_between(self.t_min_pcm, self.delta_T,
                         color="green", alpha=0.18, label="ΔT  (no-PCM − PCM)")
        ax2.set_ylabel("Temperature Reduction ΔT  [K]", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        ax2.set_ylim(bottom=0)
        ax2.spines["right"].set_visible(True)

        # Combined legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

        fig.tight_layout()
        self._save(fig, "fig3_max_battery_temp.png")

    # ── Figure 4: Heat absorbed by PCM ───────────────────────────────────────

    def _fig4_heat_absorbed(self) -> None:
        """Total, sensible, latent heat in PCM and battery energy [kJ/m]."""
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(self.t_min_pcm, self.pcm.Q_total_pcm / 1e3,
                color=C_TOTAL,    label="Total PCM heat (Q_total)")
        ax.plot(self.t_min_pcm, self.pcm.Q_sensible  / 1e3,
                color=C_SENSIBLE, label="Sensible heat", linestyle="--")
        ax.plot(self.t_min_pcm, self.pcm.Q_latent    / 1e3,
                color=C_LATENT,   label="Latent heat",   linestyle="-.")
        ax.plot(self.t_min_pcm, self.pcm.E_generated / 1e3,
                color=C_ENERGY,   label="Battery energy generated",
                linestyle=":",    linewidth=1.8)

        ax.set_xlabel("Time  [min]")
        ax.set_ylabel("Energy  [kJ / m of depth]")
        ax.set_title("Heat Absorbed by PCM vs Time")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self._save(fig, "fig4_heat_absorbed.png")

    # ── Figure 5: Liquid fraction vs time ────────────────────────────────────

    def _fig5_liquid_fraction(self) -> None:
        """Volume-averaged PCM liquid fraction vs time."""
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self.t_min_pcm, self.pcm.liq_fraction * 100,
                color=C_LIQ, label="Liquid fraction")
        ax.fill_between(self.t_min_pcm, 0, self.pcm.liq_fraction * 100,
                        color=C_LIQ, alpha=0.15)

        ax.set_xlabel("Time  [min]")
        ax.set_ylabel("Liquid Fraction  [%]")
        ax.set_title("PCM Liquid Fraction vs Time")
        ax.set_ylim(-2, 105)
        ax.axhline(100, color="grey", linestyle=":", linewidth=1.0)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        self._save(fig, "fig5_liquid_fraction.png")

    # ── Public API ────────────────────────────────────────────────────────────

    def plot_all(self) -> None:
        """Generate and save all five figures."""
        print("[postprocess] Generating figures …")
        self._fig1_battery_temperature()
        self._fig2_pcm_temperature()
        self._fig3_max_battery_temperature()
        self._fig4_heat_absorbed()
        self._fig5_liquid_fraction()
        print(f"[postprocess] All figures saved → {self.outdir}")

    def export_csv(self) -> None:
        """
        Export all time-series data to results.csv.

        The no-PCM data are interpolated onto the PCM time grid so that
        all columns share the same time axis.
        """
        csv_path = self.outdir / "results.csv"

        header = [
            "time_s",
            "T_bat_max_PCM_degC",   "T_bat_avg_PCM_degC",
            "T_pcm_avg_degC",
            "T_bat_max_noPCM_degC", "T_bat_avg_noPCM_degC",
            "delta_T_bat_max_K",
            "liquid_fraction",
            "Q_latent_kJ_per_m",    "Q_sensible_kJ_per_m",
            "Q_total_PCM_kJ_per_m",
            "E_generated_kJ_per_m",
        ]

        rows = list(zip(
            self.pcm.times,
            self.pcm.T_bat_max - 273.15,
            self.pcm.T_bat_avg - 273.15,
            self.pcm.T_pcm_avg - 273.15,
            self._nop_max_interp - 273.15,
            self._nop_avg_interp - 273.15,
            self.delta_T,
            self.pcm.liq_fraction,
            self.pcm.Q_latent    / 1e3,
            self.pcm.Q_sensible  / 1e3,
            self.pcm.Q_total_pcm / 1e3,
            self.pcm.E_generated / 1e3,
        ))

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rows:
                writer.writerow([f"{v:.6g}" for v in row])

        print(f"[postprocess] CSV exported → {csv_path}  ({len(rows)} rows)")

    def print_summary(self) -> None:
        """Print a formatted summary of the key simulation results."""
        pcm   = self.pcm
        nopcm = self.nopcm

        T_max_pcm   = pcm.T_bat_max[-1]   - 273.15
        T_max_nopcm = nopcm.T_bat_max[-1] - 273.15
        delta_T_fin = T_max_nopcm - T_max_pcm

        t_h = pcm.times[-1] / 3600.0

        bar = "═" * 60
        print(f"\n{bar}")
        print("  SIMULATION SUMMARY  –  4-cell 26650 Pack with PCM")
        print(bar)
        print(f"  Simulation duration           : {t_h:.2f} h "
              f"({pcm.times[-1]:.0f} s)")
        print(f"  Time step                     : {self.pcm.times[1]:.0f} s")
        print()
        print("  Battery temperatures (final)")
        print(f"    Max  – with PCM             : {T_max_pcm:.2f} °C")
        print(f"    Max  – no  PCM (air)        : {T_max_nopcm:.2f} °C")
        print(f"    Temperature reduction ΔT    : {delta_T_fin:.2f} K")
        print()
        print("  PCM energy storage (final, per metre of depth)")
        print(f"    Latent  heat Q_lat          : "
              f"{pcm.Q_latent[-1]/1e3:.4f} kJ/m")
        print(f"    Sensible heat Q_sen         : "
              f"{pcm.Q_sensible[-1]/1e3:.4f} kJ/m")
        print(f"    Total PCM absorption Q_tot  : "
              f"{pcm.Q_total_pcm[-1]/1e3:.4f} kJ/m")
        print(f"    Battery energy generated    : "
              f"{pcm.E_generated[-1]/1e3:.4f} kJ/m")
        print(f"    PCM absorption fraction     : "
              f"{100*pcm.Q_total_pcm[-1]/max(pcm.E_generated[-1], 1):.1f} %")
        print()
        print("  PCM state (final)")
        print(f"    Average temperature         : "
              f"{pcm.T_pcm_avg[-1] - 273.15:.2f} °C")
        print(f"    Liquid fraction             : "
              f"{pcm.liq_fraction[-1] * 100:.2f} %")
        print(bar + "\n")

    # ── Internal helper ───────────────────────────────────────────────────────

    def _save(self, fig: "plt.Figure", name: str) -> None:
        """Save a figure to output_dir and close it."""
        path = self.outdir / name
        fig.savefig(str(path), bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")
