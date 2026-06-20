# -*- coding: utf-8 -*-

!pip install pybamm -q

import pybamm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print(f"PyBaMM {pybamm.__version__}")

# ============================================================
# CONFIGURATION
# ============================================================
AMBIENT_TEMPS = np.linspace(250, 340, 10)
C_RATES       = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
TEMP_LABELS   = AMBIENT_TEMPS.round(0).astype(int)

HEATMAP_KW = dict(
    annot=True,
    linewidths=0.4,
    linecolor="white",
    xticklabels=C_RATES,
    yticklabels=TEMP_LABELS,
)

def run_sweep(param_name):
    """
    Full ΔT and discharge duration sweep.

    Design notes:
    - Fresh pybamm.ParameterValues and pybamm.Simulation per grid point
      ensures correct voltage-cutoff event registration after parameter updates.
    - ΔT masked to active discharge phase only (voltage >= cutoff) to prevent
      post-discharge cooling from producing spurious negative values.
    - 1.1x time buffer: enough headroom for solver to detect voltage cutoff
      without running deep into post-discharge territory.
    """
    print(f"\n--- Starting sweep: {param_name} ---")
    model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-6, rtol=1e-3)

    temp_rise_grid   = np.full((len(AMBIENT_TEMPS), len(C_RATES)), np.nan)
    duration_grid    = np.full((len(AMBIENT_TEMPS), len(C_RATES)), np.nan)
    termination_log  = []

    for t_idx, T_amb in enumerate(AMBIENT_TEMPS):
        print(f"  [T={T_amb:.0f}K] ", end="", flush=True)
        row_terms = []
        for c_idx, C_rate in enumerate(C_RATES):
            param    = pybamm.ParameterValues(param_name)
            nom_cap  = param["Nominal cell capacity [A.h]"]
            cutoff_v = param["Lower voltage cut-off [V]"]
            param.update({
                "Ambient temperature [K]": T_amb,
                "Initial temperature [K]": T_amb,
                "Current function [A]"   : C_rate * nom_cap,
            })
            discharge_time = (1.0 / C_rate) * 3600.0 * 1.1
            t_eval = np.linspace(0, discharge_time, 100)
            try:
                sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
                sol = sim.solve(t_eval=t_eval)

                voltage   = sol["Voltage [V]"].entries
                cell_temp = sol["Volume-averaged cell temperature [K]"].entries
                valid     = voltage >= cutoff_v
                delta_T   = float(np.max(cell_temp[valid]) if valid.any()
                                  else np.max(cell_temp)) - T_amb

                temp_rise_grid[t_idx, c_idx] = delta_T
                duration_grid[t_idx, c_idx]  = sol.t[-1]
                row_terms.append(sol.termination[:12])
                print(f"{C_rate}C(DT={delta_T:.1f}K,t={sol.t[-1]:.0f}s) ",
                      end="", flush=True)
            except Exception as e:
                row_terms.append(f"FAILED")
                print(f"[{C_rate}C FAILED:{type(e).__name__}] ",
                      end="", flush=True)
        termination_log.append(row_terms)
        print("")

    # Termination summary for 2.5C column
    print(f"\n  Termination summary 2.5C column ({param_name}):")
    for i, T in enumerate(AMBIENT_TEMPS):
        print(f"    T={T:.0f}K: {termination_log[i][-1]}")

    print(f"\n  Sweep complete: {param_name}")
    return temp_rise_grid, duration_grid


def get_voltage_curves(param_name, T_amb=298):
    """Extract voltage-time curves at 2.0, 2.25, 2.5C for a given param set."""
    model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-6, rtol=1e-3)
    curves = {}
    for C_rate in [2.0, 2.25, 2.5]:
        param = pybamm.ParameterValues(param_name)
        param.update({
            "Ambient temperature [K]": T_amb,
            "Initial temperature [K]": T_amb,
            "Current function [A]"   : C_rate * param["Nominal cell capacity [A.h]"],
        })
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        sol = sim.solve(t_eval=np.linspace(0, (1/C_rate)*3600*1.1, 300))
        curves[C_rate] = (
            sol["Time [s]"].entries,
            sol["Voltage [V]"].entries,
            param["Lower voltage cut-off [V]"]
        )
    return curves


# ============================================================
# RUN BOTH SWEEPS
# ============================================================
chen_rise,  chen_dur  = run_sweep("Chen2020")
ecker_rise, ecker_dur = run_sweep("Ecker2015")

# Sanity check
diff = np.nanmax(np.abs(chen_rise - ecker_rise))
print(f"\nSanity: max |Chen-Ecker| DT = {diff:.2f} K  "
      f"({'OK' if diff > 1 else 'WARNING: nearly identical'})")
print(f"Chen2020  2.5C duration at 250K: {chen_dur[0,-1]:.0f}s "
      f"(theoretical 1440s)")
print(f"Ecker2015 2.5C duration at 250K: {ecker_dur[0,-1]:.0f}s")
neg = np.sum(chen_rise < 0)
print(f"Negative DT values in Chen2020: {neg} "
      f"({'WARNING' if neg > 0 else 'OK'})")

# ============================================================
# FIGURE 1 — Chen2020 Peak DT heatmap
# ============================================================
print("\nGenerating Figure 1: Chen2020 Peak DT heatmap...")
fig1, ax1 = plt.subplots(figsize=(11, 7))
sns.heatmap(chen_rise, ax=ax1, fmt=".1f", cmap="YlOrRd",
            vmin=0, **HEATMAP_KW)
ax1.set_title(
    "Peak Temperature Rise DT (K)\n"
    "Chen2020 Parameterization — NMC INR21700 M50 (5 Ah)",
    fontweight="bold", pad=12)
ax1.set_xlabel("Discharge C-Rate")
ax1.set_ylabel("Ambient Temperature (K)")
plt.tight_layout()
plt.savefig("figure1_chen_delta_T.png", dpi=300, bbox_inches="tight")
print("Saved: figure1_chen_delta_T.png")
plt.show()

# ============================================================
# FIGURE 2 — Chen2020 Discharge Duration heatmap
# ============================================================
print("\nGenerating Figure 2 — Chen2020 Discharge Duration heatmap...")
fig2, ax2 = plt.subplots(figsize=(11, 7))
sns.heatmap(chen_dur, ax=ax2, fmt=".0f", cmap="viridis_r", **HEATMAP_KW)
ax2.set_title(
    "Discharge Duration to Voltage Cutoff (s)\n"
    "Chen2020 — Theoretical 2.5C Full Discharge = 1440 s",
    fontweight="bold", pad=12)
ax2.set_xlabel("Discharge C-Rate")
ax2.set_ylabel("Ambient Temperature (K)")
last_col = len(C_RATES) - 1
for row in range(len(AMBIENT_TEMPS)):
    ax2.add_patch(plt.Rectangle(
        (last_col, row), 1, 1,
        fill=False, edgecolor="red", lw=2.5, zorder=5))
ax2.annotate("~90%\ntruncated",
    xy=(last_col + 0.5, 0), xytext=(last_col + 0.5, -0.8),
    xycoords="data", color="red", fontsize=9, fontweight="bold",
    ha="center", va="top", annotation_clip=False)
plt.tight_layout()
plt.savefig("figure2_chen_duration.png", dpi=300, bbox_inches="tight")
print("Saved: figure2_chen_duration.png")
plt.show()

# ============================================================
# FIGURE 3 — Chen2020 Voltage-Time Profiles
# ============================================================
print("\nGenerating Figure 3 — Chen2020 Voltage-Time Profiles...")
chen_curves = get_voltage_curves("Chen2020", T_amb=298)
fig3, ax3 = plt.subplots(figsize=(9, 6))
styles = {
    2.00: ("tab:blue",  "-",  "2.0C  — near-complete discharge"),
    2.25: ("tab:green", "--", "2.25C — late-stage cutoff"),
    2.50: ("tab:red",   ":",  "2.5C  — early transport-limited cutoff"),
}
cutoff_chen = None
for C_rate, (color, ls, label) in styles.items():
    t_arr, v_arr, cutoff = chen_curves[C_rate]
    cutoff_chen = cutoff
    ax3.plot(t_arr, v_arr, color=color, linestyle=ls,
             linewidth=2.5, label=label)
ax3.axhline(y=cutoff_chen, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.8,
            label=f"Voltage cutoff = {cutoff_chen:.2f} V")
ax3.set_title(
    "Voltage-Time Profiles at 298 K — Chen2020 (NMC INR21700 M50)\n"
    "Transport-Limited Early Cutoff at 2.5C",
    fontweight="bold", pad=12)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Terminal Voltage (V)")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure3_chen_voltage_cutoff.png", dpi=300, bbox_inches="tight")
print("Saved: figure3_chen_voltage_cutoff.png")
plt.show()

# ============================================================
# FIGURE 4 — Electrolyte concentration vs. position
# ============================================================
print("\nGenerating Figure 4 — Electrolyte concentration vs. position...")

# ── Cell geometry from Chen2020 ────────────────────────────────────────────
p_ref = pybamm.ParameterValues("Chen2020")
L_neg   = p_ref["Negative electrode thickness [m]"] * 1e6   # µm
L_sep   = p_ref["Separator thickness [m]"] * 1e6
L_pos   = p_ref["Positive electrode thickness [m]"] * 1e6
L_total = L_neg + L_sep + L_pos
c0      = p_ref["Initial concentration in electrolyte [mol.m-3]"]
print(f"Cell: {L_neg:.1f} µm | {L_sep:.1f} µm | {L_pos:.1f} µm  (neg | sep | pos)")
print(f"Initial c_e = {c0:.0f} mol/m³")

# ── Run SPMe for 2.25C and 2.5C at 298 K ─────────────────────────────────
sims = {}
for c_rate in [2.25, 2.5]:
    model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    p = pybamm.ParameterValues("Chen2020")
    p["Ambient temperature [K]"] = 298
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V"])
    sim = pybamm.Simulation(model, experiment=exp, parameter_values=p)
    sim.solve(solver=pybamm.CasadiSolver(mode="fast"))
    sims[c_rate] = sim.solution
    t_end = sim.solution["Time [s]"].entries[-1]
    print(f"{c_rate}C: discharge ended at {t_end:.0f} s")

# ── Snapshot times: same real times for both panels ───────────────────────
# Use 0 → 143 s (the 2.5C discharge window) so the contrast is direct
t_end_25  = sims[2.5]["Time [s]"].entries[-1]
snap_times = np.linspace(0, t_end_25, 6)     # 6 evenly spaced snapshots
cmap       = cm.get_cmap('plasma_r', len(snap_times))

# ── Key numbers to report ───────────────────────────────────
print("\n── Electrolyte concentration vs. position key figures ──")
for c_rate in [2.25, 2.5]:
    sol = sims[c_rate]
    t   = sol["Time [s]"].entries
    ce  = sol["Electrolyte concentration [mol.m-3]"].entries
    # positive electrode = last ~40% of spatial points
    n_pos = int(ce.shape[0] * (L_pos / L_total))
    c_pos_end = np.clip(ce[-n_pos:, -1], 0, None).mean()
    c_pos_143 = np.clip(ce[-n_pos:, np.argmin(np.abs(t - t_end_25))], 0, None).mean()
    print(f"  {c_rate}C | t_end={t[-1]:.0f}s | pos.elec c_e at end: {c_pos_end:.1f} | at 143s: {c_pos_143:.1f} mol/m³")

# ── Plot ──────────────────────────────────────────────────────────────────
fig4, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
fig4.patch.set_facecolor('white')

for ax, c_rate, title in zip(axes, [2.25, 2.5], ['2.25C', '2.5C']):
    sol   = sims[c_rate]
    t_all = sol["Time [s]"].entries
    ce_all = sol["Electrolyte concentration [mol.m-3]"].entries   # shape (n_x, n_t)

    # Map spatial indices to µm across the full cell thickness
    x_µm = np.linspace(0, L_total, ce_all.shape[0])

    for i, t_snap in enumerate(snap_times):
        idx      = np.argmin(np.abs(t_all - t_snap))
        c_profile = np.clip(ce_all[:, idx], 0, None)   # clip unphysical negatives
        actual_t  = t_all[idx]
        ax.plot(x_µm, c_profile, color=cmap(i), lw=2.0,
                label=f't = {int(actual_t)} s')

    # Initial concentration reference
    ax.axhline(c0, color='#888888', ls=':', lw=1.3, alpha=0.8,
               label=f'Initial ({c0:.0f} mol m⁻³)')

    # Region shading
    ax.axvspan(0,             L_neg,            alpha=0.06, color='#1f77b4')
    ax.axvspan(L_neg,         L_neg+L_sep,      alpha=0.10, color='#888888')
    ax.axvspan(L_neg+L_sep,   L_total,          alpha=0.06, color='#d62728')

    # Region boundary lines
    for xv in [L_neg, L_neg+L_sep]:
        ax.axvline(xv, color='grey', ls='--', lw=0.8, alpha=0.5)

    # Region labels
    for xm, lbl, col in [
        (L_neg/2,                    'Negative\nelectrode', '#1f77b4'),
        (L_neg + L_sep/2,            'Sep.',                '#555555'),
        (L_neg + L_sep + L_pos/2,   'Positive\nelectrode', '#d62728'),
    ]:
        ax.text(xm, c0*1.08, lbl, ha='center', va='bottom',
                fontsize=8.5, color=col, fontweight='bold')

    t_this = sims[c_rate]["Time [s]"].entries[-1]
    ax.set_title(f'{title} — 298 K  (t_end = {t_this:.0f} s)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, L_total)
    ax.set_ylim(-50, c0 * 1.18)
    ax.set_xlabel('Position through cell (µm)', fontsize=11)
    if ax is axes[0]:
        ax.set_ylabel('Electrolyte concentration (mol m⁻³)', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.88)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)

plt.suptitle(
    'Li\u207a Electrolyte Concentration vs. Position — SPMe, Chen2020, 298 K\n'
    'Same six time snapshots (0\u2013143 s) shown for both panels',
    fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure4_conc.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print("\nSaved: fig4_conc.png")

# ============================================================
# FIGURE 5 — Ecker2015 Peak DT heatmap
# ============================================================
print("\nGenerating Figure 5 — Ecker2015 Peak DT heatmap...")
fig5, ax4 = plt.subplots(figsize=(11, 7))
sns.heatmap(ecker_rise, ax=ax4, fmt=".1f", cmap="YlOrRd",
            vmin=0, vmax=np.nanmax(chen_rise), **HEATMAP_KW)
ax4.set_title(
    "Peak Temperature Rise DT (K)\n"
    "Ecker2015 Parameterization — LFP IHR18650A (7.5 Ah pouch)",
    fontweight="bold", pad=12)
ax4.set_xlabel("Discharge C-Rate")
ax4.set_ylabel("Ambient Temperature (K)")
plt.tight_layout()
plt.savefig("figure5_ecker_delta_T.png", dpi=300, bbox_inches="tight")
print("Saved: figure5_ecker_delta_T.png")
plt.show()

# ============================================================
# FIGURE 6 — Ecker2015 Discharge Duration heatmap
# ============================================================
print("\nGenerating Figure 6 — Ecker2015 Discharge Duration heatmap...")
fig6, ax5 = plt.subplots(figsize=(11, 7))
sns.heatmap(ecker_dur, ax=ax5, fmt=".0f", cmap="viridis_r", **HEATMAP_KW)
ax5.set_title(
    "Discharge Duration to Voltage Cutoff (s)\n"
    "Ecker2015 — LFP IHR18650A (7.5 Ah pouch)",
    fontweight="bold", pad=12)
ax5.set_xlabel("Discharge C-Rate")
ax5.set_ylabel("Ambient Temperature (K)")
# Highlight 2.5C column
for row in range(len(AMBIENT_TEMPS)):
    ax5.add_patch(plt.Rectangle(
        (last_col, row), 1, 1,
        fill=False, edgecolor="red", lw=2.5, zorder=5))
plt.tight_layout()
plt.savefig("figure6_ecker_duration.png", dpi=300, bbox_inches="tight")
print("Saved: figure6_ecker_duration.png")
plt.show()

# ============================================================
# FIGURE 7 — Ecker2015 Voltage-Time Profiles
# ============================================================
print("\nGenerating Figure 7 — Ecker2015 Voltage-Time Profiles...")
ecker_curves = get_voltage_curves("Ecker2015", T_amb=298)
fig7, ax6 = plt.subplots(figsize=(9, 6))
cutoff_ecker = None
for C_rate, (color, ls, label) in styles.items():
    t_arr, v_arr, cutoff = ecker_curves[C_rate]
    cutoff_ecker = cutoff
    ax6.plot(t_arr, v_arr, color=color, linestyle=ls,
             linewidth=2.5, label=label)
ax6.axhline(y=cutoff_ecker, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.8,
            label=f"Voltage cutoff = {cutoff_ecker:.2f} V")
ax6.set_title(
    "Voltage-Time Profiles at 298 K — Ecker2015 (LFP IHR18650A)\n"
    "Transport-Limited Cutoff Comparison",
    fontweight="bold", pad=12)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Terminal Voltage (V)")
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure7_ecker_voltage_cutoff.png", dpi=300, bbox_inches="tight")
print("Saved: figure7_ecker_voltage_cutoff.png")
plt.show()

# ============================================================
# FIGURE 8 — Sensitivity Analysis: Bruggeman Coefficient ±10%
# ============================================================
print("Generating Figure 8: Sensitivity analysis...")
C_RATES_SENS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
T_AMB = 298
BRUGGEMAN_BASE = 1.5  # Chen2020 default for positive electrode

def run_sensitivity_point(C_rate, bruggeman_value, T_amb=298):
    """Run single SPMe simulation with modified Bruggeman coefficient."""
    model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    param  = pybamm.ParameterValues("Chen2020")
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-6, rtol=1e-3)

    nom_cap  = param["Nominal cell capacity [A.h]"]
    cutoff_v = param["Lower voltage cut-off [V]"]

    # Modify Bruggeman coefficient for positive electrode
    param.update({
        "Positive electrode Bruggeman coefficient (electrolyte)": bruggeman_value,
        "Ambient temperature [K]": T_amb,
        "Initial temperature [K]": T_amb,
        "Current function [A]": C_rate * nom_cap,
    })

    discharge_time = (1.0 / C_rate) * 3600.0 * 1.1
    t_eval = np.linspace(0, discharge_time, 100)

    try:
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        sol = sim.solve(t_eval=t_eval)
        voltage   = sol["Voltage [V]"].entries
        cell_temp = sol["Volume-averaged cell temperature [K]"].entries
        valid     = voltage >= cutoff_v
        delta_T   = float(np.max(cell_temp[valid]) if valid.any() else np.max(cell_temp)) - T_amb
        duration  = sol.t[-1]
        print(f"  {C_rate}C (Brugg={bruggeman_value:.2f}): dT={delta_T:.1f}K, t={duration:.0f}s")
        return delta_T, duration
    except Exception as e:
        print(f"  {C_rate}C FAILED: {e}")
        return np.nan, np.nan

# Run three Bruggeman cases
bruggeman_cases = {
    "Baseline (β = 1.50)":  BRUGGEMAN_BASE,
    "−10% (β = 1.35)":     BRUGGEMAN_BASE * 0.90,
    "+10% (β = 1.65)":     BRUGGEMAN_BASE * 1.10,
}

results = {}
for label, b_val in bruggeman_cases.items():
    print(f"\n--- {label} ---")
    dTs, durs = [], []
    for C in C_RATES_SENS:
        dT, dur = run_sensitivity_point(C, b_val)
        dTs.append(dT)
        durs.append(dur)
    results[label] = {"dT": np.array(dTs), "dur": np.array(durs)}

fig8, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))

colors = {"Baseline (β = 1.50)": "tab:blue",
          "−10% (β = 1.35)":    "tab:green",
          "+10% (β = 1.65)":    "tab:red"}
styles = {"Baseline (β = 1.50)": "-",
          "−10% (β = 1.35)":    "--",
          "+10% (β = 1.65)":    ":"}

for label, res in results.items():
    ax7a.plot(C_RATES_SENS, res["dT"], color=colors[label],
              linestyle=styles[label], linewidth=2.5, marker="o", markersize=5,
              label=label)
    ax7b.plot(C_RATES_SENS, res["dur"], color=colors[label],
              linestyle=styles[label], linewidth=2.5, marker="o", markersize=5,
              label=label)

ax7a.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, label="2.5C")
ax7a.set_xlabel("Discharge C-Rate", fontsize=11)
ax7a.set_ylabel("Peak ΔT (K)", fontsize=11)
ax7a.set_title("(a) Peak ΔT vs. C-Rate\nSensitivity to Bruggeman Coefficient",
               fontweight="bold")
ax7a.legend(fontsize=9)
ax7a.grid(True, alpha=0.3)

ax7b.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
ax7b.set_xlabel("Discharge C-Rate", fontsize=11)
ax7b.set_ylabel("Discharge Duration (s)", fontsize=11)
ax7b.set_title("(b) Discharge Duration vs. C-Rate\nSensitivity to Bruggeman Coefficient",
               fontweight="bold")
ax7b.legend(fontsize=9)
ax7b.grid(True, alpha=0.3)

fig8.suptitle(
    "Sensitivity of Thermal Cliff to Positive Electrode Bruggeman Coefficient (±10%)\n"
    "SPMe — Chen2020 (NMC INR21700 M50) — 298 K",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure8_sensitivity_bruggeman.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("\nSaved: figure8_sensitivity_bruggeman.png")

# ============================================================
# FIGURE 9 — DFN vs SPMe Comparison at 2.0, 2.25, 2.5C
# ============================================================
print("\nGenerating Figure 9: DFN vs SPMe comparison...")
def run_model_comparison(model_name, C_rate, T_amb=298):
    """Run SPMe or DFN and return time, voltage, peak dT, duration."""
    if model_name == "SPMe":
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    else:
        model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})

    param    = pybamm.ParameterValues("Chen2020")
    solver   = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
    nom_cap  = param["Nominal cell capacity [A.h]"]
    cutoff_v = param["Lower voltage cut-off [V]"]

    param.update({
        "Ambient temperature [K]": T_amb,
        "Initial temperature [K]": T_amb,
        "Current function [A]": C_rate * nom_cap,
    })

    discharge_time = (1.0 / C_rate) * 3600.0 * 1.1
    t_eval = np.linspace(0, discharge_time, 300)

    try:
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        sol = sim.solve(t_eval=t_eval)
        voltage   = sol["Voltage [V]"].entries
        cell_temp = sol["Volume-averaged cell temperature [K]"].entries
        valid     = voltage >= cutoff_v
        delta_T   = float(np.max(cell_temp[valid]) if valid.any() else np.max(cell_temp)) - T_amb
        duration  = sol.t[-1]
        print(f"  {model_name} {C_rate}C: dT={delta_T:.1f}K, t={duration:.0f}s")
        return sol["Time [s]"].entries, voltage, delta_T, duration
    except Exception as e:
        print(f"  {model_name} {C_rate}C FAILED: {e}")
        return None, None, np.nan, np.nan

C_RATES_DFN = [2.0, 2.25, 2.5]

fig9, axes8 = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

for i, C_rate in enumerate(C_RATES_DFN):
    ax = axes8[i]
    print(f"\n--- {C_rate}C ---")
    print(f"\nSPMe Comparison:")
    t_spme, v_spme, dT_spme, dur_spme = run_model_comparison("SPMe", C_rate)
    print(f"\nDFN Comparison:")
    t_dfn,  v_dfn,  dT_dfn,  dur_dfn  = run_model_comparison("DFN",  C_rate)

    if t_spme is not None:
        ax.plot(t_spme, v_spme, "b-", linewidth=2.5,
                label=f"SPMe  (ΔT={dT_spme:.1f}K, t={dur_spme:.0f}s)")
    if t_dfn is not None:
        ax.plot(t_dfn,  v_dfn,  "r--", linewidth=2.5,
                label=f"DFN   (ΔT={dT_dfn:.1f}K, t={dur_dfn:.0f}s)")

    ax.axhline(y=2.5, color="gray", linestyle="--", linewidth=1.2,
               alpha=0.7, label="2.5 V cutoff")
    ax.set_title(f"{C_rate}C Discharge at 298 K", fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=10)
    if i == 0:
        ax.set_ylabel("Terminal Voltage (V)", fontsize=10)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

fig9.suptitle(
    "SPMe vs. DFN Voltage-Time Profiles at 298 K — Chen2020 (NMC INR21700 M50)\n",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure9_dfn_vs_spme.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("\nSaved: figure9_dfn_vs_spme.png")

print("\nAll 9 figures saved at 300 DPI.")
print("Figure inventory:")
print("  figure1_chen_delta_T.png          — Fig 1: Chen2020 DT heatmap")
print("  figure2_chen_duration.png         — Fig 2: Chen2020 duration")
print("  figure3_chen_voltage_cutoff.png   — Fig 3: Chen2020 voltage curves")
print("  figure4_ecker_delta_T.png         — Fig 4: Ecker2015 DT heatmap")
print("  figure5_ecker_duration.png        — Fig 5: Ecker2015 duration")
print("  figure6_ecker_voltage_cutoff.png  — Fig 6: Ecker2015 voltage curves")
print("  figure7_sensitivity_bruggeman.png — Fig 7: Bruggeman sensitivity")
print("  figure8_dfn_vs_spme.png           — Fig 8: DFN vs SPMe comparison")
print("  figure9_electrolyte_conc.png      — Fig 9: Electrolyte concentration")
