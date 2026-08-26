# If running in Jupyter/Colab and pybamm is not yet installed, uncomment:
#!pip install pybamm -q

import math
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


def load_parameter_values(param_name):
    """
    Safely loads parameter values.

    FIX (Prada2013 thermal parameters): Prada2013 has no native "Cell volume",
    "Cell cooling surface area", material density/specific-heat, or current
    collector keys, because it is a positive-electrode-only (LFP) parameter
    contribution meant to be combined with a donor cell.

    The previous version filled every missing key from Marquis2019, an
    unrelated ~0.68 Ah pouch cell. Marquis2019's cooling-area-to-volume ratio
    (A/V ~ 7295 m^-1) is about 33x higher than Chen2020's real 21700 cell
    (A/V ~ 219 m^-1). Applying that geometry to a 2.3 Ah A123 cell gave the
    lumped thermal model far more cooling capacity per unit thermal mass than
    a real A123 cell has, which suppressed computed DeltaT and is the most
    likely explanation for the small, non-monotonic DeltaT values in the
    original Figure 10/12 (values mostly under 5 K, no clean trend with
    C-rate).

    Fix, in two parts:
      1. Missing material / current-collector / thermal-mass keys are now
         sourced from Chen2020 first, not Marquis2019. This keeps the whole
         parameter set on one consistent lineage: PyBaMM's own Prada2013 set
         already borrows its negative electrode and separator parameters
         from Chen2020 (see PyBaMM parameter set documentation), so donating
         the remaining thermal-mass keys from the same source avoids mixing
         in a third, unrelated cell design. Any key present in neither
         Prada2013 nor Chen2020 falls back to Marquis2019 as a last resort,
         and this is printed explicitly (once) so the approximation is
         auditable rather than silent.
      2. "Cell volume [m3]" and "Cell cooling surface area [m2]" are NOT
         borrowed from any donor cell at all. They are computed directly
         from the real geometry of the A123 ANR26650 cell that Prada2013 /
         Lain2019 parameterize (26 mm diameter x 65 mm height cylinder),
         using V = pi*r^2*h and A = 2*pi*r*h + 2*pi*r^2 (full cylinder
         surface, ends included). This exact formula, applied to Chen2020's
         own 21700 dimensions (21 mm x 70 mm), reproduces Chen2020's actual
         published Cell volume (2.4245e-5 m3 vs. published 2.42e-5 m3) and
         Cell cooling surface area (5.3109e-3 m2 vs. published 5.31e-3 m2)
         to within rounding. That cross-check confirms this is the same
         methodology already used to build Chen2020's own thermal geometry,
         so applying it to the A123 cell is on consistent footing rather
         than an ad hoc fix.

    Residual limitation: the positive-electrode
    (LFP) density and specific heat capacity are still approximated from
    Chen2020's NMC positive electrode, because no built-in PyBaMM parameter
    set ships LFP-specific values for these. This is a smaller-magnitude
    approximation (material densities typically vary by less than a factor
    of ~2) than the 33x geometric mismatch this fix removes, but it is not
    eliminated, and any quantitative DeltaT comparison against Chen2020
    should note it.
    """
    param = pybamm.ParameterValues(param_name)

    if param_name == "Prada2013":
        donor          = pybamm.ParameterValues("Chen2020")     # primary donor: same lineage as Prada2013's own anode/separator
        fallback_donor = pybamm.ParameterValues("Marquis2019")  # last resort only, for anything Chen2020 also lacks

        missing_keys = [
            # --- Geometric & Electrical (current collectors) ---
            "Negative current collector thickness [m]",
            "Positive current collector thickness [m]",
            "Negative current collector conductivity [S.m-1]",
            "Positive current collector conductivity [S.m-1]",

            # --- Lumped thermal boundary condition ---
            "Total heat transfer coefficient [W.m-2.K-1]",

            # --- Densities ---
            "Negative current collector density [kg.m-3]",
            "Positive current collector density [kg.m-3]",
            "Negative electrode density [kg.m-3]",
            "Positive electrode density [kg.m-3]",
            "Separator density [kg.m-3]",

            # --- Specific heat capacities ---
            "Negative current collector specific heat capacity [J.kg-1.K-1]",
            "Positive current collector specific heat capacity [J.kg-1.K-1]",
            "Negative electrode specific heat capacity [J.kg-1.K-1]",
            "Positive electrode specific heat capacity [J.kg-1.K-1]",
            "Separator specific heat capacity [J.kg-1.K-1]",

            # --- Thermal conductivities (needed for the Biot number check) ---
            "Negative electrode thermal conductivity [W.m-1.K-1]",
            "Positive electrode thermal conductivity [W.m-1.K-1]",
            "Separator thermal conductivity [W.m-1.K-1]",
        ]

        provenance = {}
        for key in missing_keys:
            if key not in param:
                if key in donor:
                    param[key] = donor[key]
                    provenance[key] = "Chen2020"
                else:
                    param[key] = fallback_donor[key]
                    provenance[key] = "Marquis2019 (fallback - not in Chen2020 either)"

        # --- Cell-specific geometry: computed from real A123 ANR26650 dimensions, not borrowed ---
        r_cell, h_cell = 0.013, 0.065  # 26650 format: 26 mm diameter x 65 mm height
        V_cell = math.pi * r_cell**2 * h_cell
        A_cell = 2 * math.pi * r_cell * h_cell + 2 * math.pi * r_cell**2
        param["Cell volume [m3]"] = V_cell
        param["Cell cooling surface area [m2]"] = A_cell
        provenance["Cell volume [m3]"] = f"computed from A123 ANR26650 geometry = {V_cell:.4e} m3"
        provenance["Cell cooling surface area [m2]"] = f"computed from A123 ANR26650 geometry = {A_cell:.4e} m2"

        if not load_parameter_values._provenance_printed:
            chen_check = pybamm.ParameterValues("Chen2020")
            chen_AV = chen_check["Cell cooling surface area [m2]"] / chen_check["Cell volume [m3]"]
            print("\n  [Prada2013 parameter provenance - printed once]")
            for k, v in provenance.items():
                print(f"    {k:60s} <- {v}")
            print(f"    Resulting Prada2013 A/V ratio = {A_cell / V_cell:.1f} m^-1 "
                  f"(cf. Chen2020's own A/V = {chen_AV:.1f} m^-1)\n")
            load_parameter_values._provenance_printed = True

    return param


load_parameter_values._provenance_printed = False


def run_sweep(param_name):
    """
    Full DeltaT and discharge duration sweep.

    Design notes:
    - Fresh pybamm.ParameterValues and pybamm.Simulation per grid point
      ensures correct voltage-cutoff event registration after parameter updates.
    - DeltaT masked to active discharge phase only (voltage >= cutoff) to prevent
      post-discharge cooling from producing spurious negative values.
    - 1.1x time buffer: enough headroom for solver to detect voltage cutoff
      without running deep into post-discharge territory.
    """
    print(f"\n--- Starting sweep: {param_name} ---")
    model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7)

    temp_rise_grid   = np.full((len(AMBIENT_TEMPS), len(C_RATES)), np.nan)
    duration_grid    = np.full((len(AMBIENT_TEMPS), len(C_RATES)), np.nan)
    termination_log  = []

    for t_idx, T_amb in enumerate(AMBIENT_TEMPS):
        print(f"  [T={T_amb:.0f}K] ", end="", flush=True)
        row_terms = []
        for c_idx, C_rate in enumerate(C_RATES):
            try:
                param    = load_parameter_values(param_name)
                nom_cap  = param["Nominal cell capacity [A.h]"]
                cutoff_v = param["Lower voltage cut-off [V]"]
                param.update({
                    "Ambient temperature [K]": T_amb,
                    "Initial temperature [K]": T_amb,
                    "Current function [A]"   : C_rate * nom_cap,
                })
                discharge_time = (1.0 / C_rate) * 3600.0 * 1.1
                t_eval = np.linspace(0, discharge_time, 100)

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
                row_terms.append(f"FAILED:{e}")
                print(f"[{C_rate}C FAILED:{type(e).__name__}:{e}] ",
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
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7)
    curves = {}
    for C_rate in [2.0, 2.25, 2.5]:
        param = load_parameter_values(param_name)
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
# RUN SWEEPS (CHEN, ECKER, AND PRADA)
# ============================================================
chen_rise,  chen_dur  = run_sweep("Chen2020")
ecker_rise, ecker_dur = run_sweep("Ecker2015")
prada_rise, prada_dur = run_sweep("Prada2013")

# Sanity check
diff = np.nanmax(np.abs(chen_rise - ecker_rise))
print(f"\nSanity: max |Chen-Ecker| DT = {diff:.2f} K  "
      f"({'OK' if diff > 1 else 'WARNING: nearly identical'})")
print(f"Chen2020  2.5C duration at 250K: {chen_dur[0,-1]:.0f}s (theoretical 1440s)")
print(f"Ecker2015 2.5C duration at 250K: {ecker_dur[0,-1]:.0f}s")
print(f"Prada2013 2.5C duration at 250K: {prada_dur[0,-1]:.0f}s")
neg = np.sum(chen_rise < 0)
print(f"Negative DT values in Chen2020: {neg} "
      f"({'WARNING' if neg > 0 else 'OK'})")
print(f"\nPrada2013 peak DT across full grid: "
      f"min={np.nanmin(prada_rise):.2f}K, max={np.nanmax(prada_rise):.2f}K")

# ============================================================
# FIGURE 1 — Chen2020 Peak DT heatmap
# ============================================================
print("\nGenerating Figure 1: Chen2020 Peak DT heatmap...")
fig1, ax1 = plt.subplots(figsize=(11, 7))
sns.heatmap(chen_rise, ax=ax1, fmt=".1f", cmap="YlOrRd",
            vmin=0, **HEATMAP_KW)
ax1.set_title(
    "Peak Temperature Rise (ΔT, K)\n"
    "Chen2020, SPMe",
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
    "Chen2020, SPMe",
    fontweight="bold", pad=12)
ax2.set_xlabel("Discharge C-Rate")
ax2.set_ylabel("Ambient Temperature (K)")
last_col = len(C_RATES) - 1
for row in range(len(AMBIENT_TEMPS)):
    ax2.add_patch(plt.Rectangle(
        (last_col, row), 1, 1,
        fill=False, edgecolor="red", lw=2.5, zorder=5))
ax2.text(0.01, 0.99, "Theoretical 2.5C full discharge: 1440 s",
    transform=ax2.transAxes, ha="left", va="top", fontsize=9,
    style="italic", color="dimgray")
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
styles_chen = {
    2.00: ("tab:blue",  "-",  "2.0C: near-complete discharge"),
    2.25: ("tab:green", "--", "2.25C: late-stage cutoff"),
    2.50: ("tab:red",   ":",  "2.5C: early transport-limited cutoff"),
}
cutoff_chen = None
for C_rate, (color, ls, label) in styles_chen.items():
    t_arr, v_arr, cutoff = chen_curves[C_rate]
    cutoff_chen = cutoff
    ax3.plot(t_arr, v_arr, color=color, linestyle=ls,
             linewidth=2.5, label=label)
ax3.axhline(y=cutoff_chen, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.8,
            label=f"Voltage cutoff = {cutoff_chen:.2f} V")
ax3.set_title(
    "Terminal Voltage vs. Time at 298 K\n"
    "Chen2020, SPMe",
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
# FIGURE 4 — Electrolyte concentration vs. position (Chen2020)
# ============================================================
print("\nGenerating Figure 4 — Electrolyte concentration vs. position...")

p_ref = pybamm.ParameterValues("Chen2020")
L_neg   = p_ref["Negative electrode thickness [m]"] * 1e6   # µm
L_sep   = p_ref["Separator thickness [m]"] * 1e6
L_pos   = p_ref["Positive electrode thickness [m]"] * 1e6
L_total = L_neg + L_sep + L_pos
c0      = p_ref["Initial concentration in electrolyte [mol.m-3]"]
print(f"Cell: {L_neg:.1f} µm | {L_sep:.1f} µm | {L_pos:.1f} µm  (neg | sep | pos)")
print(f"Initial c_e = {c0:.0f} mol/m³")

sims = {}
for c_rate in [2.25, 2.5]:
    model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    p = pybamm.ParameterValues("Chen2020")
    p["Ambient temperature [K]"] = 298
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V"])
    sim = pybamm.Simulation(model, experiment=exp, parameter_values=p)
    sim.solve(solver=pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7))
    sims[c_rate] = sim.solution
    t_end = sim.solution["Time [s]"].entries[-1]
    print(f"{c_rate}C: discharge ended at {t_end:.0f} s")

cmap = cm.get_cmap('plasma_r', 6)

print("\n── Electrolyte concentration vs. position key figures ──")
for c_rate in [2.25, 2.5]:
    sol = sims[c_rate]
    t   = sol["Time [s]"].entries
    ce  = sol["Electrolyte concentration [mol.m-3]"].entries
    c_collector_end = max(ce[-1, -1], 0.0)
    print(f"  {c_rate}C | t_end={t[-1]:.0f}s | c_e AT current collector (true end): {c_collector_end:.1f} mol/m³")

fig4, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
fig4.patch.set_facecolor('white')

for ax, c_rate, title in zip(axes, [2.25, 2.5], ['2.25C', '2.5C']):
    sol   = sims[c_rate]
    t_all = sol["Time [s]"].entries
    ce_all = sol["Electrolyte concentration [mol.m-3]"].entries

    t_end_this = t_all[-1]
    snap_times_this = np.linspace(0, t_end_this, 6)

    x_µm = np.linspace(0, L_total, ce_all.shape[0])

    for i, t_snap in enumerate(snap_times_this):
        idx      = np.argmin(np.abs(t_all - t_snap))
        c_profile = np.clip(ce_all[:, idx], 0, None)
        actual_t  = t_all[idx]
        ax.plot(x_µm, c_profile, color=cmap(i), lw=2.0,
                label=f't = {int(actual_t)} s')

    ax.axhline(c0, color='#888888', ls=':', lw=1.3, alpha=0.8,
               label=f'Initial ({c0:.0f} mol m⁻³)')

    ax.axvspan(0,             L_neg,            alpha=0.06, color='#1f77b4')
    ax.axvspan(L_neg,         L_neg+L_sep,      alpha=0.10, color='#888888')
    ax.axvspan(L_neg+L_sep,   L_total,          alpha=0.06, color='#d62728')

    for xv in [L_neg, L_neg+L_sep]:
        ax.axvline(xv, color='grey', ls='--', lw=0.8, alpha=0.5)

    for xm, lbl, col in [
        (L_neg/2,                    'Negative\nelectrode', '#1f77b4'),
        (L_neg + L_sep/2,            'Sep.',                '#555555'),
        (L_neg + L_sep + L_pos/2,   'Positive\nelectrode', '#d62728'),
    ]:
        ax.text(xm, c0*1.08, lbl, ha='center', va='bottom',
                fontsize=8.5, color=col, fontweight='bold')

    # --- Annotate the concentration minimum (value + position) on the
    #     final snapshot, so the depletion numbers already quoted in the
    #     text are also readable directly off the figure. ---
    final_profile = np.clip(ce_all[:, np.argmin(np.abs(t_all - snap_times_this[-1]))], 0, None)
    min_idx = int(np.argmin(final_profile))
    min_val = float(final_profile[min_idx])
    min_x   = float(x_µm[min_idx])
    ax.annotate(
        f"min = {min_val:.1f} mol m⁻³\n@ x = {min_x:.0f} µm",
        xy=(min_x, min_val),
        xytext=(min_x - L_total * 0.32, c0 * 0.55),
        fontsize=8.5, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.9),
    )

    t_this = sims[c_rate]["Time [s]"].entries[-1]
    ax.set_title(f'{title} (t_end = {t_this:.0f} s)',
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
    'Electrolyte Concentration vs. Position at 298 K\n'
    'Chen2020, SPMe',
    fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure4_conc.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print("\nSaved: fig4_chen_conc.png")

# ============================================================
# FIGURE 5 — Ecker2015 Peak DT heatmap
# ============================================================
print("\nGenerating Figure 5 — Ecker2015 Peak DT heatmap...")
fig5, ax4 = plt.subplots(figsize=(11, 7))
sns.heatmap(ecker_rise, ax=ax4, fmt=".1f", cmap="YlOrRd",
            vmin=0, vmax=np.nanmax(chen_rise), **HEATMAP_KW)
ax4.set_title(
    "Peak Temperature Rise (ΔT, K)\n"
    "Ecker2015, SPMe",
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
    "Ecker2015, SPMe",
    fontweight="bold", pad=12)
ax5.set_xlabel("Discharge C-Rate")
ax5.set_ylabel("Ambient Temperature (K)")
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
styles_ecker = {
    2.00: ("tab:blue",  "-",  "2.0C: near-complete discharge"),
    2.25: ("tab:green", "--", "2.25C: late-stage cutoff"),
    2.50: ("tab:red",   ":",  "2.5C: shortened discharge"),
}
cutoff_ecker = None
for C_rate, (color, ls, label) in styles_ecker.items():
    t_arr, v_arr, cutoff = ecker_curves[C_rate]
    cutoff_ecker = cutoff
    ax6.plot(t_arr, v_arr, color=color, linestyle=ls,
             linewidth=2.5, label=label)
ax6.axhline(y=cutoff_ecker, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.8,
            label=f"Voltage cutoff = {cutoff_ecker:.2f} V")
ax6.set_title(
    "Terminal Voltage vs. Time at 298 K\n"
    "Ecker2015, SPMe",
    fontweight="bold", pad=12)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Terminal Voltage (V)")
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure7_ecker_voltage_cutoff.png", dpi=300, bbox_inches="tight")
print("Saved: figure7_ecker_voltage_cutoff.png")
plt.show()

# ======================================================================
# FIGURE 8 — Sensitivity Analysis: Bruggeman Coefficient ±10% (Chen2020)
# ======================================================================
print("Generating Figure 8: Sensitivity analysis...")
C_RATES_SENS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
T_AMB = 298
BRUGGEMAN_BASE = 1.5

def run_sensitivity_point(C_rate, bruggeman_value, T_amb=298):
    model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    param  = load_parameter_values("Chen2020")
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7)

    nom_cap  = param["Nominal cell capacity [A.h]"]
    cutoff_v = param["Lower voltage cut-off [V]"]

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
style_mappings = {"Baseline (β = 1.50)": "-",
                  "−10% (β = 1.35)":    "--",
                  "+10% (β = 1.65)":    ":"}

for label, res in results.items():
    ax7a.plot(C_RATES_SENS, res["dT"], color=colors[label],
              linestyle=style_mappings[label], linewidth=2.5, marker="o", markersize=5,
              label=label)
    ax7b.plot(C_RATES_SENS, res["dur"], color=colors[label],
              linestyle=style_mappings[label], linewidth=2.5, marker="o", markersize=5,
              label=label)

ax7a.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, label="2.5C")
ax7a.set_xlabel("Discharge C-Rate", fontsize=11)
ax7a.set_ylabel("Peak ΔT (K)", fontsize=11)
ax7a.set_title("(a) Peak ΔT vs. C-Rate",
               fontweight="bold")
ax7a.legend(fontsize=9)
ax7a.grid(True, alpha=0.3)

ax7b.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
ax7b.set_xlabel("Discharge C-Rate", fontsize=11)
ax7b.set_ylabel("Discharge Duration (s)", fontsize=11)
ax7b.set_title("(b) Discharge Duration vs. C-Rate",
               fontweight="bold")
ax7b.legend(fontsize=9)
ax7b.grid(True, alpha=0.3)

fig8.suptitle(
    "Bruggeman Coefficient Sensitivity (±10%)\n"
    "Chen2020, SPMe",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure8_sensitivity_bruggeman.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("\nSaved: figure8_sensitivity_bruggeman.png")

# ============================================================
# FIGURE 9 — DFN vs SPMe Comparison at 2.0, 2.25, 2.5C (Chen2020)
# ============================================================
print("\nGenerating Figure 9: DFN vs SPMe comparison...")
def run_model_comparison(model_name, C_rate, T_amb=298):
    if model_name == "SPMe":
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    else:
        model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})

    param    = load_parameter_values("Chen2020")
    solver   = pybamm.CasadiSolver(mode="safe", atol=1e-9, rtol=1e-7)
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
    "Terminal Voltage vs. Time at 298 K\n"
    "Chen2020, SPMe and DFN",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure9_dfn_vs_spme.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("\nSaved: figure9_dfn_vs_spme.png")

# ======================================
# FIGURE 10 — Prada2013 Peak DT heatmap
# ======================================
print("\nGenerating Figure 10 — Prada2013 Peak DT heatmap...")
fig10, axprada_1 = plt.subplots(figsize=(11, 7))
sns.heatmap(prada_rise, ax=axprada_1, fmt=".1f", cmap="YlOrRd",
            vmin=0, vmax=np.nanmax(prada_rise), **HEATMAP_KW)
axprada_1.set_title(
    "Peak Temperature Rise (ΔT, K)\n"
    "Prada2013, SPMe",
    fontweight="bold", pad=12)
axprada_1.set_xlabel("Discharge C-Rate")
axprada_1.set_ylabel("Ambient Temperature (K)")
plt.tight_layout()
plt.savefig("figure10_prada_delta_T.png", dpi=300, bbox_inches="tight")
print("Saved: figure10_prada_delta_T.png")
plt.show()

# ============================================================
# FIGURE 11 — Prada2013 Discharge Duration heatmap
# ============================================================
print("\nGenerating Figure 11 — Prada2013 Discharge Duration heatmap...")
fig11, axprada_2 = plt.subplots(figsize=(11, 7))
sns.heatmap(prada_dur, ax=axprada_2, fmt=".0f", cmap="viridis_r", **HEATMAP_KW)
axprada_2.set_title(
    "Discharge Duration to Voltage Cutoff (s)\n"
    "Prada2013, SPMe",
    fontweight="bold", pad=12)
axprada_2.set_xlabel("Discharge C-Rate")
axprada_2.set_ylabel("Ambient Temperature (K)")
for row in range(len(AMBIENT_TEMPS)):
    axprada_2.add_patch(plt.Rectangle(
        (last_col, row), 1, 1,
        fill=False, edgecolor="red", lw=2.5, zorder=5))
plt.tight_layout()
plt.savefig("figure11_prada_duration.png", dpi=300, bbox_inches="tight")
print("Saved: figure11_prada_duration.png")
plt.show()

# ============================================================
# FIGURE 12 — Prada2013 Voltage-Time Profiles
# ============================================================
print("\nGenerating Figure 12 — Prada2013 Voltage-Time Profiles...")
prada_curves = get_voltage_curves("Prada2013", T_amb=298)
fig12, axprada_3 = plt.subplots(figsize=(9, 6))
styles_prada = {
    2.00: ("tab:blue",  "-",  "2.0C: near-complete discharge"),
    2.25: ("tab:green", "--", "2.25C: late-stage cutoff"),
    2.50: ("tab:red",   ":",  "2.5C: shortened discharge"),
}
cutoff_prada = None
for C_rate, (color, ls, label) in styles_prada.items():
    t_arr, v_arr, cutoff = prada_curves[C_rate]
    cutoff_prada = cutoff
    axprada_3.plot(t_arr, v_arr, color=color, linestyle=ls,
             linewidth=2.5, label=label)
axprada_3.axhline(y=cutoff_prada, color="gray", linestyle="--",
            linewidth=1.5, alpha=0.8,
            label=f"Voltage cutoff = {cutoff_prada:.2f} V")
axprada_3.set_title(
    "Terminal Voltage vs. Time at 298 K\n"
    "Prada2013, SPMe",
    fontweight="bold", pad=12)
axprada_3.set_xlabel("Time (s)")
axprada_3.set_ylabel("Terminal Voltage (V)")
axprada_3.legend(fontsize=10)
axprada_3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure12_prada_voltage_cutoff.png", dpi=300, bbox_inches="tight")
print("Saved: figure12_prada_voltage_cutoff.png")
plt.show()

# ============================================================
# Prada2013 Extended High-C-Rate Sweep (298 K only, 0.5-4.0C)
# Computed BEFORE Figure 13 so the concentration-profile figure can pick
# rates informed by where truncation is actually worst.
# ============================================================
print("\nRunning Prada2013 extended high-C-rate sweep (298 K, 0.5-4.0C)...")

def run_prada_extended_point(C_rate, T_amb=298):
    model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7)
    param  = load_parameter_values("Prada2013")

    nom_cap  = param["Nominal cell capacity [A.h]"]
    cutoff_v = param["Lower voltage cut-off [V]"]
    param.update({
        "Ambient temperature [K]": T_amb,
        "Initial temperature [K]": T_amb,
        "Current function [A]"   : C_rate * nom_cap,
    })

    discharge_time = (1.0 / C_rate) * 3600.0 * 1.1
    t_eval = np.linspace(0, discharge_time, 100)
    theoretical = (1.0 / C_rate) * 3600.0

    try:
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        sol = sim.solve(t_eval=t_eval)
        voltage   = sol["Voltage [V]"].entries
        cell_temp = sol["Volume-averaged cell temperature [K]"].entries
        valid     = voltage >= cutoff_v
        delta_T   = float(np.max(cell_temp[valid]) if valid.any() else np.max(cell_temp)) - T_amb
        duration  = sol.t[-1]
        print(f"  Prada2013 {C_rate:.2f}C: dT={delta_T:.2f}K, t={duration:.0f}s "
              f"({100*duration/theoretical:.0f}% of theoretical {theoretical:.0f}s)")
        return delta_T, duration, theoretical
    except Exception as e:
        print(f"  Prada2013 {C_rate:.2f}C FAILED: {e}")
        return np.nan, np.nan, theoretical

C_RATES_PRADA_EXT = np.round(np.arange(0.5, 4.0001, 0.25), 2)  # 0.5 to 4.0C, 0.25C steps

prada_ext_dT, prada_ext_dur, prada_ext_theo = [], [], []
for C in C_RATES_PRADA_EXT:
    dT, dur, theo = run_prada_extended_point(C)
    prada_ext_dT.append(dT)
    prada_ext_dur.append(dur)
    prada_ext_theo.append(theo)

prada_ext_dT    = np.array(prada_ext_dT)
prada_ext_dur   = np.array(prada_ext_dur)
prada_ext_theo  = np.array(prada_ext_theo)
prada_ext_frac  = prada_ext_dur / prada_ext_theo  # fraction of theoretical capacity delivered

# Find the single biggest one-step drop in delivered-capacity fraction: the
# best candidate for "where a cliff, if any, is starting to appear."
frac_drop     = -np.diff(prada_ext_frac)
steepest_idx  = int(np.argmax(frac_drop)) + 1
cliff_rate    = C_RATES_PRADA_EXT[steepest_idx]
cliff_drop    = frac_drop[steepest_idx - 1]

print(f"\n  Largest single-step drop in delivered-capacity fraction: "
      f"{C_RATES_PRADA_EXT[steepest_idx-1]:.2f}C -> {cliff_rate:.2f}C "
      f"({prada_ext_frac[steepest_idx-1]:.2f} -> {prada_ext_frac[steepest_idx]:.2f} "
      f"of theoretical capacity, drop = {cliff_drop:.2f})")
if cliff_drop > 0.05:
    print(f"  -> Treating {cliff_rate:.2f}C as the truncation-onset candidate for Figure 13.")
    conc_high_rate = float(cliff_rate)
else:
    print(f"  -> No step exceeds a 5-percentage-point drop up to 4.0C; "
          f"using 4.0C (highest tested rate) as the comparison point for Figure 13 instead.")
    conc_high_rate = 4.0

# ============================================================
# FIGURE 13 — Prada2013 electrolyte concentration vs. position
# Mirrors Figure 4, comparing 2.5C (main-grid ceiling) against
# the truncation-onset candidate identified above.
# ============================================================
print(f"\nGenerating Figure 13 — Prada2013 electrolyte concentration vs. position "
      f"(2.5C vs {conc_high_rate:.2f}C)...")

p_prada_ref = load_parameter_values("Prada2013")
Lp_neg   = p_prada_ref["Negative electrode thickness [m]"] * 1e6
Lp_sep   = p_prada_ref["Separator thickness [m]"] * 1e6
Lp_pos   = p_prada_ref["Positive electrode thickness [m]"] * 1e6
Lp_total = Lp_neg + Lp_sep + Lp_pos
c0_prada = p_prada_ref["Initial concentration in electrolyte [mol.m-3]"]
cutoff_v_prada = p_prada_ref["Lower voltage cut-off [V]"]
print(f"Prada2013 cell: {Lp_neg:.1f} µm | {Lp_sep:.1f} µm | {Lp_pos:.1f} µm  (neg | sep | pos)")
print(f"Initial c_e = {c0_prada:.0f} mol/m³, cutoff = {cutoff_v_prada:.2f} V")

prada_conc_rates = [2.5, conc_high_rate]
sims_prada = {}
for c_rate in prada_conc_rates:
    model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    p = load_parameter_values("Prada2013")
    p["Ambient temperature [K]"] = 298
    exp = pybamm.Experiment([f"Discharge at {c_rate}C until {cutoff_v_prada}V"])
    sim = pybamm.Simulation(model, experiment=exp, parameter_values=p)
    sim.solve(solver=pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7))
    sims_prada[c_rate] = sim.solution
    t_end = sim.solution["Time [s]"].entries[-1]
    print(f"{c_rate}C: discharge ended at {t_end:.0f} s")

cmap_prada = cm.get_cmap('plasma_r', 6)

fig13, axes13 = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
fig13.patch.set_facecolor('white')

for ax, c_rate, title in zip(axes13, prada_conc_rates,
                              [f'{prada_conc_rates[0]}C', f'{prada_conc_rates[1]:.2f}C']):
    sol   = sims_prada[c_rate]
    t_all = sol["Time [s]"].entries
    ce_all = sol["Electrolyte concentration [mol.m-3]"].entries

    t_end_this = t_all[-1]
    snap_times_this = np.linspace(0, t_end_this, 6)

    x_um = np.linspace(0, Lp_total, ce_all.shape[0])

    for i, t_snap in enumerate(snap_times_this):
        idx      = np.argmin(np.abs(t_all - t_snap))
        c_profile = np.clip(ce_all[:, idx], 0, None)
        actual_t  = t_all[idx]
        ax.plot(x_um, c_profile, color=cmap_prada(i), lw=2.0,
                label=f't = {int(actual_t)} s')

    ax.axhline(c0_prada, color='#888888', ls=':', lw=1.3, alpha=0.8,
               label=f'Initial ({c0_prada:.0f} mol m⁻³)')

    ax.axvspan(0,                Lp_neg,             alpha=0.06, color='#1f77b4')
    ax.axvspan(Lp_neg,           Lp_neg+Lp_sep,      alpha=0.10, color='#888888')
    ax.axvspan(Lp_neg+Lp_sep,    Lp_total,           alpha=0.06, color='#d62728')

    for xv in [Lp_neg, Lp_neg+Lp_sep]:
        ax.axvline(xv, color='grey', ls='--', lw=0.8, alpha=0.5)

    for xm, lbl, col in [
        (Lp_neg/2,                     'Negative\nelectrode', '#1f77b4'),
        (Lp_neg + Lp_sep/2,            'Sep.',                '#555555'),
        (Lp_neg + Lp_sep + Lp_pos/2,   'Positive\nelectrode', '#d62728'),
    ]:
        ax.text(xm, c0_prada*1.08, lbl, ha='center', va='bottom',
                fontsize=8.5, color=col, fontweight='bold')

    t_this = sims_prada[c_rate]["Time [s]"].entries[-1]
    ax.set_title(f'{title} (t_end = {t_this:.0f} s)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, Lp_total)
    ax.set_ylim(-50, c0_prada * 1.18)
    ax.set_xlabel('Position through cell (µm)', fontsize=11)
    if ax is axes13[0]:
        ax.set_ylabel('Electrolyte concentration (mol m⁻³)', fontsize=11)
    ax.legend(fontsize=8.5, loc='lower left', framealpha=0.88)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=10)

plt.suptitle(
    'Electrolyte Concentration vs. Position at 298 K\n'
    'Prada2013, SPMe',
    fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure13_prada_conc.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
plt.close()
print("\nSaved: figure13_prada_conc.png")

# ==========================================================================
# FIGURE 14 — Prada2013 extended high-C-rate sweep plot (298 K, 0.5-4.0C)
# ==========================================================================
print("\nGenerating Figure 14 — Prada2013 extended high-C-rate sweep...")
fig14, (ax14a, ax14b) = plt.subplots(1, 2, figsize=(14, 6))

ax14a.plot(C_RATES_PRADA_EXT, prada_ext_dT, color="tab:red", marker="o",
           markersize=5, linewidth=2.2, label="Prada2013")
ax14a.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, label="2.5C (main-grid ceiling)")
ax14a.set_xlabel("Discharge C-Rate", fontsize=11)
ax14a.set_ylabel("Peak ΔT (K)", fontsize=11)
ax14a.set_title("(a) Peak ΔT vs. C-Rate", fontweight="bold")
ax14a.legend(fontsize=9)
ax14a.grid(True, alpha=0.3)

ax14b.plot(C_RATES_PRADA_EXT, prada_ext_dur, color="tab:red", marker="o",
           markersize=5, linewidth=2.2, label="Actual duration")
ax14b.plot(C_RATES_PRADA_EXT, prada_ext_theo, color="black", linestyle=":",
           linewidth=1.5, label="Theoretical full discharge")
ax14b.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
ax14b.set_xlabel("Discharge C-Rate", fontsize=11)
ax14b.set_ylabel("Discharge Duration (s)", fontsize=11)
ax14b.set_title("(b) Discharge Duration vs. C-Rate", fontweight="bold")
ax14b.legend(fontsize=9)
ax14b.grid(True, alpha=0.3)

fig14.suptitle(
    "Extended High C-Rate Sweep\n"
    "Prada2013, SPMe",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure14_prada_extended_sweep.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("\nSaved: figure14_prada_extended_sweep.png")

print("\nAll main-text figures saved successfully.")

# ============================================================================
# EXTENSION HELPERS
# ============================================================================
# The sections below add supplementary robustness checks raised in review:
#   Fig 15 - cutoff voltage & initial electrolyte concentration sensitivity
#   Fig 16 - additional transport-parameter sensitivity (diffusivity,
#            conductivity, transference number, separator tortuosity)
#   Fig 17 - Bruggeman coefficient sensitivity widened to +/-10/20/30%
#   Fig 18 - heat source decomposition (ohmic / reaction / reversible),
#            SPMe vs DFN, at 2.25C and 2.5C
#   Fig 19 - capacity-normalized peak heat generation, Chen2020 vs Ecker2015
#   Fig 20 - OCP-swap counterfactual (Chen2020 cell with Prada2013's
#            positive-electrode OCP) to separate chemistry-shape effects
#            from transport effects
#   + a DFN mesh-refinement convergence check (printed, Section "DFN
#     CONVERGENCE CHECK") and a Biot number estimate (printed, Section
#     "BIOT NUMBER ESTIMATE") for all three parameter sets.
# None of this touches the main-grid sweeps or Figures 1-14 above; every
# section below runs its own small, targeted simulations at 298 K only,
# using the existing load_parameter_values() / C_RATES_SENS machinery.
# ============================================================================

def run_point(param_name, C_rate, T_amb=298, overrides=None, model_cls=None,
              var_pts=None, solver_mode="fast"):
    """
    Shared single-point runner used by all extension sections below.
    Mirrors run_sensitivity_point()'s logic (Figure 8) but accepts an
    arbitrary dict of parameter overrides and an optional model/solver
    swap, so it can be reused for transport-parameter sweeps, cutoff/
    initial-concentration sweeps, OCP swaps, and DFN mesh refinement
    without duplicating the solve/DeltaT/duration bookkeeping each time.
    Returns (delta_T, duration, solution_or_None).
    """
    model_cls = model_cls or pybamm.lithium_ion.SPMe
    model  = model_cls(options={"thermal": "lumped"})
    param  = load_parameter_values(param_name)
    solver = pybamm.CasadiSolver(mode=solver_mode, atol=1e-9, rtol=1e-7)

    nom_cap  = param["Nominal cell capacity [A.h]"]
    if overrides:
        param.update(overrides)
    cutoff_v = param["Lower voltage cut-off [V]"]

    param.update({
        "Ambient temperature [K]": T_amb,
        "Initial temperature [K]": T_amb,
        "Current function [A]": C_rate * nom_cap,
    })

    discharge_time = (1.0 / C_rate) * 3600.0 * 1.1
    t_eval = np.linspace(0, discharge_time, 100)

    try:
        sim_kwargs = dict(parameter_values=param, solver=solver)
        if var_pts is not None:
            sim_kwargs["var_pts"] = var_pts
        sim = pybamm.Simulation(model, **sim_kwargs)
        sol = sim.solve(t_eval=t_eval)
        voltage   = sol["Voltage [V]"].entries
        cell_temp = sol["Volume-averaged cell temperature [K]"].entries
        valid     = voltage >= cutoff_v
        delta_T   = float(np.max(cell_temp[valid]) if valid.any() else np.max(cell_temp)) - T_amb
        duration  = sol.t[-1]
        return delta_T, duration, sol
    except Exception as e:
        print(f"    FAILED ({param_name}, {C_rate}C): {type(e).__name__}: {e}")
        return np.nan, np.nan, None


def scale_function_parameter(original_func, scale_factor):
    """Wrap a callable PyBaMM parameter (e.g. electrolyte diffusivity/
    conductivity, both functions of (c_e, T)) so its output is scaled by
    a constant factor, regardless of the wrapped function's own signature."""
    def scaled(*args, **kwargs):
        return scale_factor * original_func(*args, **kwargs)
    return scaled


# ============================================================================
# FIGURE 15 - Cutoff Voltage & Initial Electrolyte Concentration Sensitivity
#             (Chen2020, 298 K)
# ============================================================================
print("\nGenerating Figure 15: cutoff voltage & initial electrolyte "
      "concentration sensitivity...")

BASE_CUTOFF = pybamm.ParameterValues("Chen2020")["Lower voltage cut-off [V]"]
BASE_C0     = pybamm.ParameterValues("Chen2020")["Initial concentration in electrolyte [mol.m-3]"]

cutoff_cases = {
    "Baseline (2.50 V)": BASE_CUTOFF,
    "-0.10 V (2.40 V)":  BASE_CUTOFF - 0.10,
    "+0.10 V (2.60 V)":  BASE_CUTOFF + 0.10,
}
c0_cases = {
    "Baseline (1000 mol m⁻³)": BASE_C0,
    "-10% (900 mol m⁻³)":      BASE_C0 * 0.90,
    "+10% (1100 mol m⁻³)":     BASE_C0 * 1.10,
}
sweep_colors = {0: "tab:blue", 1: "tab:green", 2: "tab:red"}
sweep_styles = {0: "-", 1: "--", 2: ":"}

fig15, (ax15a, ax15b) = plt.subplots(1, 2, figsize=(14, 6))

print("  -- cutoff voltage sweep --")
for i, (label, v_cut) in enumerate(cutoff_cases.items()):
    dTs = []
    for C in C_RATES_SENS:
        dT, dur, _ = run_point("Chen2020", C, overrides={"Lower voltage cut-off [V]": v_cut})
        dTs.append(dT)
        print(f"    {label} @ {C}C: dT={dT:.1f}K" if not np.isnan(dT) else f"    {label} @ {C}C: FAILED")
    ax15a.plot(C_RATES_SENS, dTs, color=sweep_colors[i], linestyle=sweep_styles[i],
               marker="o", markersize=5, linewidth=2.5, label=label)

print("  -- initial electrolyte concentration sweep --")
for i, (label, c0_val) in enumerate(c0_cases.items()):
    dTs = []
    for C in C_RATES_SENS:
        dT, dur, _ = run_point("Chen2020", C,
                                overrides={"Initial concentration in electrolyte [mol.m-3]": c0_val})
        dTs.append(dT)
        print(f"    {label} @ {C}C: dT={dT:.1f}K" if not np.isnan(dT) else f"    {label} @ {C}C: FAILED")
    ax15b.plot(C_RATES_SENS, dTs, color=sweep_colors[i], linestyle=sweep_styles[i],
               marker="o", markersize=5, linewidth=2.5, label=label)

for ax, title in [(ax15a, "(a) Cutoff Voltage Sensitivity"),
                   (ax15b, "(b) Initial Electrolyte Concentration Sensitivity")]:
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Discharge C-Rate", fontsize=11)
    ax.set_ylabel("Peak ΔT (K)", fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig15.suptitle(
    "Cutoff Voltage and Initial Electrolyte Concentration Sensitivity\n"
    "Chen2020, SPMe, 298 K",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure15_cutoff_c0_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: figure15_cutoff_c0_sensitivity.png")

# ============================================================================
# FIGURE 16 - Additional Transport-Parameter Sensitivity (+/-10%)
#             (Chen2020, 298 K)
# ============================================================================
print("\nGenerating Figure 16: additional transport-parameter sensitivity...")

p_base = pybamm.ParameterValues("Chen2020")
base_diffusivity   = p_base["Electrolyte diffusivity [m2.s-1]"]
base_conductivity  = p_base["Electrolyte conductivity [S.m-1]"]
base_transference  = p_base["Cation transference number"]
base_sep_bruggeman = p_base["Separator Bruggeman coefficient (electrolyte)"]

transport_params = {
    "Electrolyte diffusivity": {
        "key": "Electrolyte diffusivity [m2.s-1]",
        "baseline": base_diffusivity,
        "make_override": lambda f: scale_function_parameter(f, 1.0),
        "is_function": True,
        "base_value": base_diffusivity,
    },
    "Electrolyte conductivity": {
        "key": "Electrolyte conductivity [S.m-1]",
        "base_value": base_conductivity,
        "is_function": True,
    },
    "Cation transference number": {
        "key": "Cation transference number",
        "base_value": base_transference,
        "is_function": False,
    },
    "Separator Bruggeman coefficient": {
        "key": "Separator Bruggeman coefficient (electrolyte)",
        "base_value": base_sep_bruggeman,
        "is_function": False,
    },
}

fig16, axes16 = plt.subplots(2, 2, figsize=(14, 11))
axes16 = axes16.flatten()

for panel_idx, (pname, info) in enumerate(transport_params.items()):
    ax = axes16[panel_idx]
    key = info["key"]
    print(f"  -- {pname} +/-10% --")
    for i, pct in enumerate([-0.10, 0.0, 0.10]):
        label = ("Baseline" if pct == 0.0 else f"{pct:+.0%}")
        if info["is_function"]:
            base_func = info["base_value"]
            override_val = scale_function_parameter(base_func, 1.0 + pct)
        else:
            override_val = info["base_value"] * (1.0 + pct)
        dTs = []
        for C in C_RATES_SENS:
            dT, dur, _ = run_point("Chen2020", C, overrides={key: override_val})
            dTs.append(dT)
        print(f"    {label}: dT range {np.nanmin(dTs):.1f}-{np.nanmax(dTs):.1f} K")
        ax.plot(C_RATES_SENS, dTs, color=sweep_colors[i], linestyle=sweep_styles[i],
                marker="o", markersize=4, linewidth=2.2, label=label)
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Discharge C-Rate", fontsize=10)
    ax.set_ylabel("Peak ΔT (K)", fontsize=10)
    ax.set_title(f"({chr(97+panel_idx)}) {pname} ±10%", fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

fig16.suptitle(
    "Additional Transport-Parameter Sensitivity (±10%)\n"
    "Chen2020, SPMe, 298 K",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure16_transport_param_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: figure16_transport_param_sensitivity.png")

# ============================================================================
# FIGURE 17 - Bruggeman Coefficient Sensitivity Widened to +/-10/20/30%
#             (Chen2020, 298 K)
# ============================================================================
print("\nGenerating Figure 17: widened Bruggeman coefficient sensitivity...")

bruggeman_wide_cases = {
    "Baseline (β = 1.50)": 0.00,
    "-10% (β = 1.35)":     -0.10,
    "+10% (β = 1.65)":      0.10,
    "-20% (β = 1.20)":     -0.20,
    "+20% (β = 1.80)":      0.20,
    "-30% (β = 1.05)":     -0.30,
    "+30% (β = 1.95)":      0.30,
}
wide_colors = plt.cm.coolwarm(np.linspace(0, 1, len(bruggeman_wide_cases)))

fig17, (ax17a, ax17b) = plt.subplots(1, 2, figsize=(14, 6))
for i, (label, pct) in enumerate(bruggeman_wide_cases.items()):
    b_val = BRUGGEMAN_BASE * (1.0 + pct)
    dTs, durs = [], []
    print(f"  -- {label} --")
    for C in C_RATES_SENS:
        dT, dur, _ = run_point("Chen2020", C,
                                overrides={"Positive electrode Bruggeman coefficient (electrolyte)": b_val})
        dTs.append(dT)
        durs.append(dur)
    ax17a.plot(C_RATES_SENS, dTs, color=wide_colors[i],
               linestyle="-" if pct == 0 else "--", linewidth=2.2 if pct == 0 else 1.6,
               marker="o", markersize=4, label=label)
    ax17b.plot(C_RATES_SENS, durs, color=wide_colors[i],
               linestyle="-" if pct == 0 else "--", linewidth=2.2 if pct == 0 else 1.6,
               marker="o", markersize=4, label=label)

for ax, ylabel, title in [(ax17a, "Peak ΔT (K)", "(a) Peak ΔT vs. C-Rate"),
                           (ax17b, "Discharge Duration (s)", "(b) Discharge Duration vs. C-Rate")]:
    ax.axvline(x=2.5, color="gray", linestyle=":", alpha=0.5, label="2.5C")
    ax.set_xlabel("Discharge C-Rate", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=7.5, ncol=1)
    ax.grid(True, alpha=0.3)

fig17.suptitle(
    "Bruggeman Coefficient Sensitivity, Widened Range (±10/20/30%)\n"
    "Chen2020, SPMe, 298 K",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure17_bruggeman_wide_sensitivity.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: figure17_bruggeman_wide_sensitivity.png")

# ============================================================================
# FIGURE 18 - Heat Source Decomposition (Ohmic / Reaction / Reversible),
#             SPMe vs DFN, at 2.25C and 2.5C (Chen2020, 298 K)
# ============================================================================
print("\nGenerating Figure 18: heat source decomposition, SPMe vs DFN...")

HEAT_VARS = {
    "Ohmic":      "Ohmic heating [W]",
    "Reaction":   "Irreversible electrochemical heating [W]",
    "Reversible": "Reversible heating [W]",
    "Total":      "Total heating [W]",
}

def get_heat_decomposition(model_cls, C_rate, T_amb=298):
    model  = model_cls(options={"thermal": "lumped"})
    param  = load_parameter_values("Chen2020")
    solver = pybamm.CasadiSolver(mode="safe", atol=1e-9, rtol=1e-7)
    nom_cap = param["Nominal cell capacity [A.h]"]
    param.update({
        "Ambient temperature [K]": T_amb,
        "Initial temperature [K]": T_amb,
        "Current function [A]": C_rate * nom_cap,
    })
    t_eval = np.linspace(0, (1.0 / C_rate) * 3600.0 * 1.1, 200)
    try:
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        sol = sim.solve(t_eval=t_eval)
        t = sol["Time [s]"].entries
        out = {"t": t}
        for label, var in HEAT_VARS.items():
            out[label] = sol[var].entries
        return out
    except Exception as e:
        print(f"    FAILED ({model_cls.__name__}, {C_rate}C): {type(e).__name__}: {e}")
        return None

fig18, axes18 = plt.subplots(2, 2, figsize=(14, 10), sharex="col")
model_pairs = [("SPMe", pybamm.lithium_ion.SPMe), ("DFN", pybamm.lithium_ion.DFN)]
c_rates_18 = [2.25, 2.5]
heat_colors = {"Ohmic": "tab:orange", "Reaction": "tab:purple",
               "Reversible": "tab:green", "Total": "black"}

for col, C_rate in enumerate(c_rates_18):
    for row, (model_name, model_cls) in enumerate(model_pairs):
        ax = axes18[row, col]
        print(f"  -- {model_name}, {C_rate}C --")
        data = get_heat_decomposition(model_cls, C_rate)
        if data is not None:
            for label in ["Ohmic", "Reaction", "Reversible"]:
                ax.plot(data["t"], data[label], color=heat_colors[label],
                        linewidth=2.0, label=label)
            ax.plot(data["t"], data["Total"], color=heat_colors["Total"],
                    linewidth=1.4, linestyle="--", label="Total (check)")
        ax.set_title(f"{model_name}, {C_rate}C", fontweight="bold", fontsize=10.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        if row == 1:
            ax.set_xlabel("Time (s)", fontsize=10)
        if col == 0:
            ax.set_ylabel("Heat generation rate (W)", fontsize=10)

fig18.suptitle(
    "Heat Source Decomposition vs. Time (Ohmic, Reaction, Reversible)\n"
    "Chen2020, SPMe vs. DFN, 298 K",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure18_heat_decomposition.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: figure18_heat_decomposition.png")

# ============================================================================
# FIGURE 19 - Capacity-Normalized Peak Heat Generation: Chen2020 vs Ecker2015
#             (298 K)
# ============================================================================
print("\nGenerating Figure 19: capacity-normalized peak heat, Chen2020 vs "
      "Ecker2015...")

def get_peak_total_heat(param_name, C_rate, T_amb=298):
    model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    param  = load_parameter_values(param_name)
    solver = pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7)
    nom_cap = param["Nominal cell capacity [A.h]"]
    param.update({
        "Ambient temperature [K]": T_amb,
        "Initial temperature [K]": T_amb,
        "Current function [A]": C_rate * nom_cap,
    })
    t_eval = np.linspace(0, (1.0 / C_rate) * 3600.0 * 1.1, 150)
    try:
        sim = pybamm.Simulation(model, parameter_values=param, solver=solver)
        sol = sim.solve(t_eval=t_eval)
        peak_heat_W = float(np.max(sol["Total heating [W]"].entries))
        return peak_heat_W, nom_cap
    except Exception as e:
        print(f"    FAILED ({param_name}, {C_rate}C): {type(e).__name__}: {e}")
        return np.nan, np.nan

norm_results = {"Chen2020": {"raw": [], "norm": []}, "Ecker2015": {"raw": [], "norm": []}}
for pname in ["Chen2020", "Ecker2015"]:
    print(f"  -- {pname} --")
    for C in C_RATES_SENS:
        peak_W, cap_Ah = get_peak_total_heat(pname, C)
        norm_results[pname]["raw"].append(peak_W)
        norm_results[pname]["norm"].append(peak_W / cap_Ah if cap_Ah and not np.isnan(cap_Ah) else np.nan)
        print(f"    {C}C: peak={peak_W:.3f} W, capacity={cap_Ah:.3f} Ah, "
              f"normalized={peak_W/cap_Ah if cap_Ah else float('nan'):.3f} W/Ah")

fig19, (ax19a, ax19b) = plt.subplots(1, 2, figsize=(14, 6))
for pname, color in [("Chen2020", "tab:blue"), ("Ecker2015", "tab:orange")]:
    ax19a.plot(C_RATES_SENS, norm_results[pname]["raw"], color=color,
               marker="o", markersize=5, linewidth=2.2, label=pname)
    ax19b.plot(C_RATES_SENS, norm_results[pname]["norm"], color=color,
               marker="o", markersize=5, linewidth=2.2, label=pname)

ax19a.set_title("(a) Peak Total Heat Generation (Raw)", fontweight="bold")
ax19a.set_ylabel("Peak heat generation rate (W)", fontsize=11)
ax19b.set_title("(b) Peak Heat Generation, Normalized by Capacity", fontweight="bold")
ax19b.set_ylabel("Peak heat generation rate per unit capacity (W A⁻¹h⁻¹)", fontsize=11)
for ax in (ax19a, ax19b):
    ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Discharge C-Rate", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig19.suptitle(
    "Capacity-Normalized Peak Heat Generation, Chen2020 vs. Ecker2015\n"
    "SPMe, 298 K",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure19_capacity_normalized_heat.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: figure19_capacity_normalized_heat.png")

# ============================================================================
# FIGURE 20 - OCP-Swap Counterfactual: Chen2020 Cell with Prada2013's
#             Positive-Electrode OCP (298 K, 2.5C)
# ============================================================================
print("\nGenerating Figure 20: OCP-swap counterfactual (Chen2020 vs "
      "Chen2020-with-Prada2013-OCP vs Prada2013)...")
print("  NOTE: this swaps only 'Positive electrode OCP [V]'; Chen2020's own "
      "stoichiometry window (x_0/x_100) is NOT recalibrated to LFP's, so "
      "this isolates OCP *shape* only qualitatively, not a fully rebalanced "
      "LFP cell. See paper Limitations for this caveat.")

ocp_swap_C_rate = 2.5

prada_ocp = pybamm.ParameterValues("Prada2013")["Positive electrode OCP [V]"]

dT_chen, dur_chen, sol_chen = run_point("Chen2020", ocp_swap_C_rate, solver_mode="safe")
dT_swap, dur_swap, sol_swap = run_point(
    "Chen2020", ocp_swap_C_rate, solver_mode="safe",
    overrides={"Positive electrode OCP [V]": prada_ocp}
)
dT_prada, dur_prada, sol_prada = run_point("Prada2013", ocp_swap_C_rate, solver_mode="safe")

print(f"  Chen2020 (native OCP):            dT={dT_chen:.1f} K, t={dur_chen:.0f} s")
print(f"  Chen2020 (Prada2013 OCP swapped): dT={dT_swap:.1f} K, t={dur_swap:.0f} s")
print(f"  Prada2013 (native, own transport): dT={dT_prada:.1f} K, t={dur_prada:.0f} s")

fig20, ax20 = plt.subplots(figsize=(9, 6.5))
for sol, label, color, ls in [
    (sol_chen,  f"Chen2020 native (ΔT={dT_chen:.1f}K, t={dur_chen:.0f}s)",           "tab:red",   "-"),
    (sol_swap,  f"Chen2020 + Prada2013 OCP (ΔT={dT_swap:.1f}K, t={dur_swap:.0f}s)",  "tab:purple","--"),
    (sol_prada, f"Prada2013 native (ΔT={dT_prada:.1f}K, t={dur_prada:.0f}s)",        "tab:green", ":"),
]:
    if sol is not None:
        ax20.plot(sol["Time [s]"].entries, sol["Voltage [V]"].entries,
                  color=color, linestyle=ls, linewidth=2.5, label=label)

ax20.set_xlabel("Time (s)", fontsize=11)
ax20.set_ylabel("Terminal Voltage (V)", fontsize=11)
ax20.set_title(
    f"OCP-Swap Counterfactual at {ocp_swap_C_rate}C, 298 K\n"
    "Isolating OCP Shape from Transport Parameters",
    fontweight="bold"
)
ax20.legend(fontsize=8.5, loc="upper right")
ax20.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figure20_ocp_swap_counterfactual.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()
print("Saved: figure20_ocp_swap_counterfactual.png")

# ============================================================================
# DFN CONVERGENCE CHECK (printed) - default vs. refined mesh, Chen2020,
# 2.0C and 2.5C, 298 K
# ============================================================================
print("\n" + "=" * 70)
print("DFN CONVERGENCE CHECK (Chen2020, 298 K)")
print("=" * 70)

default_var_pts = pybamm.lithium_ion.DFN().default_var_pts
default_report = {k: default_var_pts[k] for k in ["x_n", "x_s", "x_p", "r_n", "r_p"]}
refined_var_pts = dict(default_var_pts)
refined_var_pts.update({"x_n": 40, "x_s": 40, "x_p": 40, "r_n": 40, "r_p": 40})
refined_report = {k: refined_var_pts[k] for k in ["x_n", "x_s", "x_p", "r_n", "r_p"]}

print(f"Default DFN discretization (through-thickness x_n/x_s/x_p, "
      f"particle r_n/r_p): {default_report}")
print(f"Refined DFN discretization for this check: {refined_report}")

for C_rate in [2.0, 2.5]:
    print(f"\n  -- {C_rate}C --")
    dT_def, dur_def, _ = run_point("Chen2020", C_rate, model_cls=pybamm.lithium_ion.DFN,
                                    solver_mode="safe")
    print(f"    Default mesh:  dT={dT_def:.2f} K, t={dur_def:.0f} s"
          if not np.isnan(dT_def) else "    Default mesh:  FAILED")
    dT_ref, dur_ref, _ = run_point("Chen2020", C_rate, model_cls=pybamm.lithium_ion.DFN,
                                    var_pts=refined_var_pts, solver_mode="safe")
    print(f"    Refined mesh:  dT={dT_ref:.2f} K, t={dur_ref:.0f} s"
          if not np.isnan(dT_ref) else "    Refined mesh:  FAILED (see note below)")
    if not np.isnan(dT_def) and not np.isnan(dT_ref):
        print(f"    Difference:    ΔT changed by {abs(dT_ref - dT_def):.2f} K "
              f"({100*abs(dT_ref-dT_def)/dT_def:.1f}%), "
              f"duration changed by {abs(dur_ref - dur_def):.0f} s")
    elif not np.isnan(dT_def) and np.isnan(dT_ref):
        print("    Refined-mesh run did not converge at this rate; this is "
              "itself informative and is reported as-is rather than silently "
              "retried at looser tolerances.")

# ============================================================================
# BIOT NUMBER ESTIMATE (printed) - Chen2020, Ecker2015, Prada2013
# ============================================================================
print("\n" + "=" * 70)
print("BIOT NUMBER ESTIMATE (all three parameter sets)")
print("=" * 70)
print("Bi = h * L_c / k, with L_c = V_cell / A_cooling (volume-to-area "
      "characteristic length) and k = an effective/representative thermal "
      "conductivity for the cell. Bi << 1 supports the lumped-thermal-mass "
      "assumption used throughout this study; Bi approaching or exceeding "
      "~0.1 signals that internal temperature gradients may not be "
      "negligible, which is a limitation of every ΔT value reported here.")

H_COEFF = 10.0  # W m^-2 K^-1, matches "Total heat transfer coefficient" used throughout

for pname in ["Chen2020", "Ecker2015", "Prada2013"]:
    p = load_parameter_values(pname)
    V_cell = p["Cell volume [m3]"]
    A_cell = p["Cell cooling surface area [m2]"]
    L_c = V_cell / A_cell

    # Representative effective thermal conductivity: volume-weighted average
    # of the three layer thermal conductivities, weighted by layer thickness
    # as a simple proxy for through-cell (radial, for a cylinder) conduction.
    try:
        k_n = p["Negative electrode thermal conductivity [W.m-1.K-1]"]
        k_s = p["Separator thermal conductivity [W.m-1.K-1]"]
        k_p = p["Positive electrode thermal conductivity [W.m-1.K-1]"]
        L_n = p["Negative electrode thickness [m]"]
        L_s = p["Separator thickness [m]"]
        L_p = p["Positive electrode thickness [m]"]
        k_eff = (k_n * L_n + k_s * L_s + k_p * L_p) / (L_n + L_s + L_p)
    except KeyError as e:
        print(f"  {pname}: missing thermal conductivity key ({e}); skipping k_eff.")
        continue

    Bi = H_COEFF * L_c / k_eff
    print(f"\n  {pname}:")
    print(f"    V_cell = {V_cell:.3e} m^3, A_cell = {A_cell:.3e} m^2, "
          f"L_c = V/A = {L_c*1e3:.3f} mm")
    print(f"    k_eff (thickness-weighted) = {k_eff:.2f} W/m/K")
    print(f"    Bi = h*L_c/k_eff = {Bi:.4f}  "
          f"({'lumped assumption reasonable' if Bi < 0.1 else 'lumped assumption questionable'})")

print("\nAll figures (main text + extensions) saved successfully.")
