"""
plot_vt_comparison.py
Dual-axis comparison chart:
  X-axis  : Time [s]
  Left Y  : Voltage [V]  -- measured (Excel) vs simulated (NTGK)
  Right Y : Temperature [C] -- measured (Excel) vs simulated (NTGK)

Run:
    python plot_vt_comparison.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── 1. Load Excel data ────────────────────────────────────────────────────────
df = pd.ExcelFile("LIB.xlsx").parse("Sheet2", header=None)
V_excel = df.iloc[:, 1].values.astype(float)   # pack voltage [V]  (4S)
T_excel = df.iloc[:, 2].values.astype(float)   # temperature [C]
t_excel = np.arange(len(V_excel), dtype=float) # 1 row = 1 second

print(f"Excel data: {len(t_excel)} points  "
      f"V: {V_excel[0]:.3f} -> {V_excel[-1]:.3f} V  "
      f"T: {T_excel[0]:.1f} -> {T_excel[-1]:.1f} C")

# ── 2. Run simulation ─────────────────────────────────────────────────────────
print("\nRunning NTGK simulation ...")
import config
from core.ntgk   import NTGKModel
from core.pack   import Pack
from core.solver import Solver

ntgk = NTGKModel("LIB.xlsx", C_nom=4.5, poly_order=5, n_cells_series=4)
ntgk.reset(dod=0.0)
t_end = config.T_END

pack = Pack(ntgk=ntgk, nx=50, ny=50, nz=50)
grid = pack.build()

# Storage for per-step V and T
sim_t   = []
sim_V   = []   # pack terminal voltage [V]
sim_T   = []   # max battery temperature [C]

def _cb(solver, step):
    s = pack.stats()
    sim_t.append(solver.time)
    sim_V.append(s["V_terminal_V"] * 4)   # per-cell -> pack
    sim_T.append(s["T_max_C"])

solver = Solver(grid, dt=config.DT, t_end=t_end,
                t_amb=config.T_AMB, h_conv=config.H_CONV,
                save_every=1)              # every step for smooth curve
solver.run(pack=pack, callback=_cb, verbose=False)

sim_t = np.array(sim_t)
sim_V = np.array(sim_V)
sim_T = np.array(sim_T)

print(f"Sim done: {len(sim_t)} points  "
      f"V: {sim_V[0]:.3f} -> {sim_V[-1]:.3f} V  "
      f"T: {sim_T[0]:.1f} -> {sim_T[-1]:.1f} C")

# ── 3. Plot ───────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Voltage (left axis)
l1, = ax1.plot(t_excel, V_excel,
               color="#1565C0", lw=2.0, alpha=0.85,
               label="Voltage — Measured (Excel)")
l2, = ax1.plot(sim_t,   sim_V,
               color="#42A5F5", lw=2.0, linestyle="--",
               label="Voltage — Simulated (NTGK)")

# Temperature (right axis)
l3, = ax2.plot(t_excel, T_excel,
               color="#B71C1C", lw=2.0, alpha=0.85,
               label="Temperature — Measured (Excel)")
l4, = ax2.plot(sim_t,   sim_T,
               color="#EF9A9A", lw=2.0, linestyle="--",
               label="Temperature — Simulated (NTGK)")

# Axes labels and formatting
ax1.set_xlabel("Time  (s)", fontsize=13)
ax1.set_ylabel("Pack Voltage  (V)", fontsize=13, color="#1565C0")
ax2.set_ylabel("Temperature  (C)", fontsize=13, color="#B71C1C")
ax1.tick_params(axis="y", labelcolor="#1565C0")
ax2.tick_params(axis="y", labelcolor="#B71C1C")

# Align zero lines nicely
ax1.set_xlim(0, max(t_excel[-1], sim_t[-1]))
ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# Legend (all 4 lines together)
lines  = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", fontsize=10,
           framealpha=0.9, edgecolor="#cccccc")

ax1.grid(True, which="major", alpha=0.3, linestyle="-")
ax1.grid(True, which="minor", alpha=0.1, linestyle=":")

ax1.set_title(
    "PyBatterySim — Voltage & Temperature vs Time\n"
    "Measured (LIB.xlsx) vs NTGK Simulation  |  4S1P · 26650 · 2.4C discharge",
    fontsize=12, fontweight="bold"
)

fig.tight_layout()
out = os.path.join(config.OUTPUT_DIR, "vt_comparison.png")
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
fig.savefig(out, dpi=180)
print(f"\nSaved -> {os.path.abspath(out)}")
plt.close(fig)
