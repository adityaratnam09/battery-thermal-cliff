#!/usr/bin/env python3
"""
Battery Safe Operating Envelope
================================

Simulates SPMe/DFN discharge behaviour across ambient-temperature x C-rate
grids for three lithium-ion parameter sets (Chen2020, Ecker2015, Prada2013)
using PyBaMM, and regenerates every figure used in the accompanying paper,
"When More Current Means Less Heat: A Transport-Limited Thermal Response in
Lithium-Ion Battery Simulations."

Usage
-----
    python generate_figures.py                  # full run, headless, saves to ./figures
    python generate_figures.py --show            # also pop up each figure as it renders
    python generate_figures.py --quick           # coarse grids, for a fast smoke test
    python generate_figures.py --skip-extensions  # main-text figures (1-14) only
    python generate_figures.py --output-dir out  # choose where figures/logs are written

See README.md for setup instructions and requirements.txt for dependencies.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import matplotlib

# Headless by default so the script can run in a plain shell / CI job;
# --show switches to an interactive backend before pyplot is imported.
if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pybamm
import seaborn as sns

logger = logging.getLogger("battery_safe_operating_envelope")


# ============================================================================
# Configuration
# ============================================================================

AMBIENT_TEMPS = np.linspace(250, 340, 10)
C_RATES = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5])
C_RATES_SENS = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0])
BRUGGEMAN_BASE = 1.5
T_ROOM = 298.0

TEMP_LABELS = AMBIENT_TEMPS.round(0).astype(int)
HEATMAP_KW = dict(
    annot=True,
    linewidths=0.4,
    linecolor="white",
    xticklabels=C_RATES,
    yticklabels=TEMP_LABELS,
)

# Populated by main() from CLI args; module-level so plotting helpers can
# reach them without threading extra parameters through every call.
OUTPUT_DIR = Path("figures")
SHOW_PLOTS = False


# ============================================================================
# Parameter loading
# ============================================================================


def load_parameter_values(param_name: str) -> pybamm.ParameterValues:
    """Load a PyBaMM parameter set, patching Prada2013's missing thermal
    and current-collector keys.

    Prada2013 ships positive-electrode (LFP) parameters only, so several
    thermal/geometric keys are missing entirely. Missing keys are sourced
    from Chen2020 first (same anode/separator lineage as Prada2013 in
    PyBaMM's own parameter sets), falling back to Marquis2019 only for
    anything Chen2020 also lacks. Cell volume and cooling surface area are
    computed directly from the real A123 ANR26650 geometry (26 mm x 65 mm
    cylinder) rather than borrowed from any donor cell. Provenance is
    printed once, on first use, for auditability. Full derivation and the
    residual LFP-density/specific-heat approximation are discussed in the
    paper's Methods and Limitations sections.
    """
    param = pybamm.ParameterValues(param_name)

    if param_name == "Prada2013":
        donor = pybamm.ParameterValues("Chen2020")
        fallback_donor = pybamm.ParameterValues("Marquis2019")

        missing_keys = [
            "Negative current collector thickness [m]",
            "Positive current collector thickness [m]",
            "Negative current collector conductivity [S.m-1]",
            "Positive current collector conductivity [S.m-1]",
            "Total heat transfer coefficient [W.m-2.K-1]",
            "Negative current collector density [kg.m-3]",
            "Positive current collector density [kg.m-3]",
            "Negative electrode density [kg.m-3]",
            "Positive electrode density [kg.m-3]",
            "Separator density [kg.m-3]",
            "Negative current collector specific heat capacity [J.kg-1.K-1]",
            "Positive current collector specific heat capacity [J.kg-1.K-1]",
            "Negative electrode specific heat capacity [J.kg-1.K-1]",
            "Positive electrode specific heat capacity [J.kg-1.K-1]",
            "Separator specific heat capacity [J.kg-1.K-1]",
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
                    provenance[key] = "Marquis2019 (fallback)"

        r_cell, h_cell = 0.013, 0.065  # 26650 format: 26 mm diameter x 65 mm height
        v_cell = math.pi * r_cell**2 * h_cell
        a_cell = 2 * math.pi * r_cell * h_cell + 2 * math.pi * r_cell**2
        param["Cell volume [m3]"] = v_cell
        param["Cell cooling surface area [m2]"] = a_cell
        provenance["Cell volume [m3]"] = f"computed from A123 ANR26650 geometry = {v_cell:.4e} m3"
        provenance["Cell cooling surface area [m2]"] = f"computed from A123 ANR26650 geometry = {a_cell:.4e} m2"

        if not load_parameter_values._provenance_printed:
            chen_check = pybamm.ParameterValues("Chen2020")
            chen_av = chen_check["Cell cooling surface area [m2]"] / chen_check["Cell volume [m3]"]
            logger.info("Prada2013 parameter provenance (printed once):")
            for k, v in provenance.items():
                logger.info("  %-60s <- %s", k, v)
            logger.info(
                "  Resulting Prada2013 A/V ratio = %.1f m^-1 (cf. Chen2020's own A/V = %.1f m^-1)",
                a_cell / v_cell, chen_av,
            )
            load_parameter_values._provenance_printed = True

    return param


load_parameter_values._provenance_printed = False


# ============================================================================
# Core simulation helpers
# ============================================================================


def simulate(
    param_name,
    c_rate,
    t_amb=T_ROOM,
    overrides=None,
    model_cls=None,
    var_pts=None,
    solver_mode="fast",
    t_points=100,
):
    """Run one constant-current discharge and return (param, solution).

    `solution` is None if the solver failed; the failure is logged rather
    than raised so batch sweeps can continue past a single bad grid point.
    """
    model_cls = model_cls or pybamm.lithium_ion.SPMe
    model = model_cls(options={"thermal": "lumped"})
    param = load_parameter_values(param_name)
    if overrides:
        param.update(overrides)

    nom_cap = param["Nominal cell capacity [A.h]"]
    param.update({
        "Ambient temperature [K]": t_amb,
        "Initial temperature [K]": t_amb,
        "Current function [A]": c_rate * nom_cap,
    })

    solver = pybamm.CasadiSolver(mode=solver_mode, atol=1e-9, rtol=1e-7)
    t_eval = np.linspace(0, (1.0 / c_rate) * 3600.0 * 1.1, t_points)

    try:
        sim_kwargs = dict(parameter_values=param, solver=solver)
        if var_pts is not None:
            sim_kwargs["var_pts"] = var_pts
        sim = pybamm.Simulation(model, **sim_kwargs)
        return param, sim.solve(t_eval=t_eval)
    except Exception as exc:  # noqa: BLE001 - batch sweeps must survive one bad point
        logger.warning("Simulation failed (%s, %.2fC): %s: %s", param_name, c_rate, type(exc).__name__, exc)
        return param, None


def peak_temp_rise_and_duration(sol, t_amb, cutoff_v):
    """Peak volume-averaged cell ΔT (masked to the active-discharge phase,
    i.e. while voltage is still above cutoff) and total discharge duration."""
    if sol is None:
        return np.nan, np.nan
    voltage = sol["Voltage [V]"].entries
    cell_temp = sol["Volume-averaged cell temperature [K]"].entries
    valid = voltage >= cutoff_v
    delta_t = float(np.max(cell_temp[valid]) if valid.any() else np.max(cell_temp)) - t_amb
    duration = sol.t[-1]
    return delta_t, duration


def run_discharge(param_name, c_rate, t_amb=T_ROOM, overrides=None, model_cls=None,
                   var_pts=None, solver_mode="fast", t_points=100):
    """Convenience wrapper: simulate() + peak_temp_rise_and_duration() in one call.
    Returns (delta_T, duration, solution)."""
    param, sol = simulate(param_name, c_rate, t_amb, overrides, model_cls, var_pts, solver_mode, t_points)
    cutoff_v = param["Lower voltage cut-off [V]"]
    delta_t, duration = peak_temp_rise_and_duration(sol, t_amb, cutoff_v)
    return delta_t, duration, sol


def run_sweep(param_name):
    """Full ΔT and discharge-duration sweep across AMBIENT_TEMPS x C_RATES."""
    logger.info("--- Starting sweep: %s ---", param_name)
    temp_rise_grid = np.full((len(AMBIENT_TEMPS), len(C_RATES)), np.nan)
    duration_grid = np.full((len(AMBIENT_TEMPS), len(C_RATES)), np.nan)

    for t_idx, t_amb in enumerate(AMBIENT_TEMPS):
        row = []
        for c_idx, c_rate in enumerate(C_RATES):
            delta_t, duration, _ = run_discharge(param_name, c_rate, t_amb)
            temp_rise_grid[t_idx, c_idx] = delta_t
            duration_grid[t_idx, c_idx] = duration
            row.append(f"{c_rate}C(dT={delta_t:.1f}K,t={duration:.0f}s)")
        logger.info("  T=%.0fK: %s", t_amb, " ".join(row))

    logger.info("Sweep complete: %s", param_name)
    return temp_rise_grid, duration_grid


def get_voltage_curves(param_name, t_amb=T_ROOM, c_rates=(2.00, 2.25, 2.50)):
    """Voltage-vs-time curves at the given C-rates for one parameter set.
    Returns {C_rate: (t, V, cutoff_V)}."""
    curves = {}
    for c_rate in c_rates:
        _, duration, sol = run_discharge(param_name, c_rate, t_amb, t_points=300)
        param = load_parameter_values(param_name)
        curves[c_rate] = (
            sol["Time [s]"].entries,
            sol["Voltage [V]"].entries,
            param["Lower voltage cut-off [V]"],
        )
    return curves


def scale_function_parameter(original_func, scale_factor):
    """Wrap a callable PyBaMM parameter (e.g. electrolyte diffusivity or
    conductivity, both functions of (c_e, T)) so its output is scaled by a
    constant factor, regardless of the wrapped function's own signature."""
    def scaled(*args, **kwargs):
        return scale_factor * original_func(*args, **kwargs)
    return scaled


# ============================================================================
# Plotting helpers
# ============================================================================


def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info("Saved: %s", path)
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_delta_t_heatmap(grid, label, filename, vmax=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(grid, ax=ax, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=vmax, **HEATMAP_KW)
    ax.set_title(f"Peak Temperature Rise (\u0394T, K)\n{label}, SPMe", fontweight="bold", pad=12)
    ax.set_xlabel("Discharge C-Rate")
    ax.set_ylabel("Ambient Temperature (K)")
    save_figure(fig, filename)


def plot_duration_heatmap(grid, label, filename, note=None, truncation_note=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(grid, ax=ax, fmt=".0f", cmap="viridis_r", **HEATMAP_KW)
    ax.set_title(f"Discharge Duration to Voltage Cutoff (s)\n{label}, SPMe", fontweight="bold", pad=12)
    ax.set_xlabel("Discharge C-Rate")
    ax.set_ylabel("Ambient Temperature (K)")

    last_col = len(C_RATES) - 1
    for row in range(len(AMBIENT_TEMPS)):
        ax.add_patch(plt.Rectangle((last_col, row), 1, 1, fill=False, edgecolor="red", lw=2.5, zorder=5))
    if note:
        ax.text(0.01, 0.99, note, transform=ax.transAxes, ha="left", va="top",
                 fontsize=9, style="italic", color="dimgray")
    if truncation_note:
        ax.annotate(truncation_note, xy=(last_col + 0.5, 0), xytext=(last_col + 0.5, -0.8),
                     xycoords="data", color="red", fontsize=9, fontweight="bold",
                     ha="center", va="top", annotation_clip=False)
    save_figure(fig, filename)


def plot_voltage_curves(curves, label, filename, style_map, subtitle="Terminal Voltage vs. Time at 298 K"):
    fig, ax = plt.subplots(figsize=(9, 6))
    cutoff = None
    for c_rate, (color, ls, curve_label) in style_map.items():
        t_arr, v_arr, cutoff = curves[c_rate]
        ax.plot(t_arr, v_arr, color=color, linestyle=ls, linewidth=2.5, label=curve_label)
    ax.axhline(y=cutoff, color="gray", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Voltage cutoff = {cutoff:.2f} V")
    ax.set_title(f"{subtitle}\n{label}, SPMe", fontweight="bold", pad=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Terminal Voltage (V)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    save_figure(fig, filename)


def plot_concentration_profiles(sims, geometry, c0, filename, label, rate_titles, annotate_min=False):
    """Electrolyte concentration vs. through-cell position, at six evenly
    spaced snapshots in time, for two C-rates side by side."""
    l_neg, l_sep, l_pos, l_total = geometry
    cmap = matplotlib.colormaps["plasma_r"].resampled(6)
    #cmap = cm.get_cmap("plasma_r", 6)
    c_rates = list(sims.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, c_rate, title in zip(axes, c_rates, rate_titles):
        sol = sims[c_rate]
        t_all = sol["Time [s]"].entries
        ce_all = sol["Electrolyte concentration [mol.m-3]"].entries
        snap_times = np.linspace(0, t_all[-1], 6)
        x_um = np.linspace(0, l_total, ce_all.shape[0])

        for i, t_snap in enumerate(snap_times):
            idx = np.argmin(np.abs(t_all - t_snap))
            profile = np.clip(ce_all[:, idx], 0, None)
            ax.plot(x_um, profile, color=cmap(i), lw=2.0, label=f"t = {int(t_all[idx])} s")

        ax.axhline(c0, color="#888888", ls=":", lw=1.3, alpha=0.8, label=f"Initial ({c0:.0f} mol m\u207b\u00b3)")
        ax.axvspan(0, l_neg, alpha=0.06, color="#1f77b4")
        ax.axvspan(l_neg, l_neg + l_sep, alpha=0.10, color="#888888")
        ax.axvspan(l_neg + l_sep, l_total, alpha=0.06, color="#d62728")
        for xv in (l_neg, l_neg + l_sep):
            ax.axvline(xv, color="grey", ls="--", lw=0.8, alpha=0.5)
        for xm, lbl, col in [
            (l_neg / 2, "Negative\nelectrode", "#1f77b4"),
            (l_neg + l_sep / 2, "Sep.", "#555555"),
            (l_neg + l_sep + l_pos / 2, "Positive\nelectrode", "#d62728"),
        ]:
            ax.text(xm, c0 * 1.08, lbl, ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")

        if annotate_min:
            final_profile = np.clip(ce_all[:, np.argmin(np.abs(t_all - snap_times[-1]))], 0, None)
            min_idx = int(np.argmin(final_profile))
            ax.annotate(
                f"min = {final_profile[min_idx]:.1f} mol m\u207b\u00b3\n@ x = {x_um[min_idx]:.0f} \u00b5m",
                xy=(x_um[min_idx], final_profile[min_idx]),
                xytext=(x_um[min_idx] - l_total * 0.32, c0 * 0.55),
                fontsize=8.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.9),
            )

        ax.set_title(f"{title} (t_end = {t_all[-1]:.0f} s)", fontsize=11, fontweight="bold")
        ax.set_xlim(0, l_total)
        ax.set_ylim(-50, c0 * 1.18)
        ax.set_xlabel("Position through cell (\u00b5m)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Electrolyte concentration (mol m\u207b\u00b3)", fontsize=11)
        ax.legend(fontsize=8.5, loc="lower left", framealpha=0.88)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle(f"Electrolyte Concentration vs. Position at 298 K\n{label}, SPMe",
                 fontsize=11, fontweight="bold", y=1.02)
    save_figure(fig, filename)


def cell_geometry(param):
    l_neg = param["Negative electrode thickness [m]"] * 1e6
    l_sep = param["Separator thickness [m]"] * 1e6
    l_pos = param["Positive electrode thickness [m]"] * 1e6
    return l_neg, l_sep, l_pos, l_neg + l_sep + l_pos


# ============================================================================
# Main-text figures (1-14)
# ============================================================================


def generate_main_figures():
    chen_rise, chen_dur = run_sweep("Chen2020")
    ecker_rise, ecker_dur = run_sweep("Ecker2015")
    prada_rise, prada_dur = run_sweep("Prada2013")

    # Sanity checks
    diff = np.nanmax(np.abs(chen_rise - ecker_rise))
    logger.info("Sanity: max |Chen-Ecker| dT = %.2f K (%s)", diff, "OK" if diff > 1 else "WARNING: nearly identical")
    logger.info("Chen2020 2.5C duration at 250K: %.0fs (theoretical 1440s)", chen_dur[0, -1])
    neg = np.sum(chen_rise < 0)
    logger.info("Negative dT values in Chen2020: %d (%s)", neg, "WARNING" if neg > 0 else "OK")

    # Figures 1-3: Chen2020
    plot_delta_t_heatmap(chen_rise, "Chen2020", "figure1_chen_delta_T.png")
    plot_duration_heatmap(chen_dur, "Chen2020", "figure2_chen_duration.png",
                           note="Theoretical 2.5C full discharge: 1440 s",
                           truncation_note="~90%\ntruncated")
    chen_curves = get_voltage_curves("Chen2020")
    plot_voltage_curves(chen_curves, "Chen2020", "figure3_chen_voltage_cutoff.png", {
        2.00: ("tab:blue", "-", "2.0C: near-complete discharge"),
        2.25: ("tab:green", "--", "2.25C: late-stage cutoff"),
        2.50: ("tab:red", ":", "2.5C: early transport-limited cutoff"),
    })

    # Figure 4: Chen2020 electrolyte concentration vs. position
    p_ref = pybamm.ParameterValues("Chen2020")
    geometry = cell_geometry(p_ref)
    c0 = p_ref["Initial concentration in electrolyte [mol.m-3]"]
    sims_chen = {}
    for c_rate in (2.25, 2.5):
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
        p = pybamm.ParameterValues("Chen2020")
        p["Ambient temperature [K]"] = T_ROOM
        exp = pybamm.Experiment([f"Discharge at {c_rate}C until 2.5V"])
        sim = pybamm.Simulation(model, experiment=exp, parameter_values=p)
        sim.solve(solver=pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7))
        sims_chen[c_rate] = sim.solution
    plot_concentration_profiles(sims_chen, geometry, c0, "figure4_chen_conc.png", "Chen2020",
                                 ["2.25C", "2.5C"], annotate_min=True)

    # Figures 5-7: Ecker2015
    plot_delta_t_heatmap(ecker_rise, "Ecker2015", "figure5_ecker_delta_T.png", vmax=np.nanmax(chen_rise))
    plot_duration_heatmap(ecker_dur, "Ecker2015", "figure6_ecker_duration.png")
    ecker_curves = get_voltage_curves("Ecker2015")
    plot_voltage_curves(ecker_curves, "Ecker2015", "figure7_ecker_voltage_cutoff.png", {
        2.00: ("tab:blue", "-", "2.0C: near-complete discharge"),
        2.25: ("tab:green", "--", "2.25C: late-stage cutoff"),
        2.50: ("tab:red", ":", "2.5C: shortened discharge"),
    })

    # Figure 8: Bruggeman coefficient sensitivity (+/-10%), Chen2020
    generate_bruggeman_sensitivity_figure()

    # Figure 9: DFN vs SPMe comparison, Chen2020
    generate_dfn_vs_spme_figure()

    # Figures 10-12: Prada2013
    plot_delta_t_heatmap(prada_rise, "Prada2013", "figure10_prada_delta_T.png", vmax=np.nanmax(prada_rise))
    plot_duration_heatmap(prada_dur, "Prada2013", "figure11_prada_duration.png")
    prada_curves = get_voltage_curves("Prada2013")
    plot_voltage_curves(prada_curves, "Prada2013", "figure12_prada_voltage_cutoff.png", {
        2.00: ("tab:blue", "-", "2.0C: near-complete discharge"),
        2.25: ("tab:green", "--", "2.25C: late-stage cutoff"),
        2.50: ("tab:red", ":", "2.5C: shortened discharge"),
    })

    # Figures 13-14: Prada2013 extended high-C-rate sweep + concentration profile
    generate_prada_extended_figures()

    logger.info("All main-text figures saved successfully.")


def generate_bruggeman_sensitivity_figure():
    logger.info("Generating Figure 8: Bruggeman coefficient sensitivity...")
    cases = {
        "Baseline (\u03b2 = 1.50)": BRUGGEMAN_BASE,
        "\u221210% (\u03b2 = 1.35)": BRUGGEMAN_BASE * 0.90,
        "+10% (\u03b2 = 1.65)": BRUGGEMAN_BASE * 1.10,
    }
    colors = {"Baseline (\u03b2 = 1.50)": "tab:blue", "\u221210% (\u03b2 = 1.35)": "tab:green", "+10% (\u03b2 = 1.65)": "tab:red"}
    styles = {"Baseline (\u03b2 = 1.50)": "-", "\u221210% (\u03b2 = 1.35)": "--", "+10% (\u03b2 = 1.65)": ":"}

    results = {}
    for label, b_val in cases.items():
        dts, durs = [], []
        for c_rate in C_RATES_SENS:
            dt, dur, _ = run_discharge(
                "Chen2020", c_rate,
                overrides={"Positive electrode Bruggeman coefficient (electrolyte)": b_val},
            )
            dts.append(dt)
            durs.append(dur)
        results[label] = {"dT": np.array(dts), "dur": np.array(durs)}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    for label, res in results.items():
        ax_a.plot(C_RATES_SENS, res["dT"], color=colors[label], linestyle=styles[label],
                  linewidth=2.5, marker="o", markersize=5, label=label)
        ax_b.plot(C_RATES_SENS, res["dur"], color=colors[label], linestyle=styles[label],
                  linewidth=2.5, marker="o", markersize=5, label=label)

    ax_a.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, label="2.5C")
    ax_a.set(xlabel="Discharge C-Rate", ylabel="Peak \u0394T (K)")
    ax_a.set_title("(a) Peak \u0394T vs. C-Rate", fontweight="bold")
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)

    ax_b.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax_b.set(xlabel="Discharge C-Rate", ylabel="Discharge Duration (s)")
    ax_b.set_title("(b) Discharge Duration vs. C-Rate", fontweight="bold")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

    fig.suptitle("Bruggeman Coefficient Sensitivity (\u00b110%)\nChen2020, SPMe", fontsize=11, fontweight="bold")
    save_figure(fig, "figure8_sensitivity_bruggeman.png")


def generate_dfn_vs_spme_figure():
    logger.info("Generating Figure 9: DFN vs SPMe comparison...")
    c_rates = [2.0, 2.25, 2.5]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)

    for i, c_rate in enumerate(c_rates):
        ax = axes[i]
        dt_spme, dur_spme, sol_spme = run_discharge("Chen2020", c_rate, solver_mode="safe", t_points=300)
        dt_dfn, dur_dfn, sol_dfn = run_discharge(
            "Chen2020", c_rate, model_cls=pybamm.lithium_ion.DFN, solver_mode="safe", t_points=300,
        )
        if sol_spme is not None:
            ax.plot(sol_spme["Time [s]"].entries, sol_spme["Voltage [V]"].entries, "b-", linewidth=2.5,
                    label=f"SPMe  (\u0394T={dt_spme:.1f}K, t={dur_spme:.0f}s)")
        if sol_dfn is not None:
            ax.plot(sol_dfn["Time [s]"].entries, sol_dfn["Voltage [V]"].entries, "r--", linewidth=2.5,
                    label=f"DFN   (\u0394T={dt_dfn:.1f}K, t={dur_dfn:.0f}s)")
        ax.axhline(y=2.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7, label="2.5 V cutoff")
        ax.set_title(f"{c_rate}C Discharge at 298 K", fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=10)
        if i == 0:
            ax.set_ylabel("Terminal Voltage (V)", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Terminal Voltage vs. Time at 298 K\nChen2020, SPMe and DFN", fontsize=11, fontweight="bold")
    save_figure(fig, "figure9_dfn_vs_spme.png")


def generate_prada_extended_figures():
    logger.info("Running Prada2013 extended high-C-rate sweep (298 K, 0.5-4.0C)...")
    c_rates_ext = np.round(np.arange(0.5, 4.0001, 0.25), 2)
    dts, durs, theos = [], [], []
    for c_rate in c_rates_ext:
        dt, dur, _ = run_discharge("Prada2013", c_rate)
        theo = (1.0 / c_rate) * 3600.0
        dts.append(dt)
        durs.append(dur)
        theos.append(theo)
        logger.info("  Prada2013 %.2fC: dT=%.2fK, t=%.0fs (%.0f%% of theoretical %.0fs)",
                     c_rate, dt, dur, 100 * dur / theo, theo)

    dts, durs, theos = np.array(dts), np.array(durs), np.array(theos)
    frac = durs / theos

    # Identify the steepest single-step drop in delivered-capacity fraction
    # as the truncation-onset candidate for the concentration-profile figure.
    frac_drop = -np.diff(frac)
    steepest_idx = int(np.argmax(frac_drop)) + 1
    cliff_drop = frac_drop[steepest_idx - 1]
    if cliff_drop > 0.05:
        conc_high_rate = float(c_rates_ext[steepest_idx])
        logger.info("Largest capacity-fraction drop at %.2fC -> using it for Figure 13.", conc_high_rate)
    else:
        conc_high_rate = 4.0
        logger.info("No drop exceeds 5 percentage points up to 4.0C; using 4.0C for Figure 13.")

    # Figure 13: concentration profile, 2.5C vs. the truncation-onset candidate
    p_ref = load_parameter_values("Prada2013")
    geometry = cell_geometry(p_ref)
    c0 = p_ref["Initial concentration in electrolyte [mol.m-3]"]
    cutoff_v = p_ref["Lower voltage cut-off [V]"]

    sims_prada = {}
    for c_rate in (2.5, conc_high_rate):
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
        p = load_parameter_values("Prada2013")
        p["Ambient temperature [K]"] = T_ROOM
        exp = pybamm.Experiment([f"Discharge at {c_rate}C until {cutoff_v}V"])
        sim = pybamm.Simulation(model, experiment=exp, parameter_values=p)
        sim.solve(solver=pybamm.CasadiSolver(mode="fast", atol=1e-9, rtol=1e-7))
        sims_prada[c_rate] = sim.solution
    plot_concentration_profiles(
        sims_prada, geometry, c0, "figure13_prada_conc.png", "Prada2013",
        ["2.5C", f"{conc_high_rate:.2f}C"],
    )

    # Figure 14: extended sweep, dT and duration vs. C-rate
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    ax_a.plot(c_rates_ext, dts, color="tab:red", marker="o", markersize=5, linewidth=2.2, label="Prada2013")
    ax_a.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5, label="2.5C (main-grid ceiling)")
    ax_a.set(xlabel="Discharge C-Rate", ylabel="Peak \u0394T (K)")
    ax_a.set_title("(a) Peak \u0394T vs. C-Rate", fontweight="bold")
    ax_a.legend(fontsize=9)
    ax_a.grid(True, alpha=0.3)

    ax_b.plot(c_rates_ext, durs, color="tab:red", marker="o", markersize=5, linewidth=2.2, label="Actual duration")
    ax_b.plot(c_rates_ext, theos, color="black", linestyle=":", linewidth=1.5, label="Theoretical full discharge")
    ax_b.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
    ax_b.set(xlabel="Discharge C-Rate", ylabel="Discharge Duration (s)")
    ax_b.set_title("(b) Discharge Duration vs. C-Rate", fontweight="bold")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3)

    fig.suptitle("Extended High C-Rate Sweep\nPrada2013, SPMe", fontsize=11, fontweight="bold")
    save_figure(fig, "figure14_prada_extended_sweep.png")


# ============================================================================
# Supplementary figures (15-20) and diagnostic checks
# ============================================================================


def generate_extension_figures():
    generate_cutoff_c0_sensitivity_figure()
    generate_transport_parameter_sensitivity_figure()
    generate_bruggeman_wide_sensitivity_figure()
    generate_heat_decomposition_figure()
    generate_capacity_normalized_heat_figure()
    generate_ocp_swap_figure()
    run_dfn_convergence_check()
    estimate_biot_numbers()


def generate_cutoff_c0_sensitivity_figure():
    logger.info("Generating Figure 15: cutoff voltage & initial electrolyte concentration sensitivity...")
    base_cutoff = pybamm.ParameterValues("Chen2020")["Lower voltage cut-off [V]"]
    base_c0 = pybamm.ParameterValues("Chen2020")["Initial concentration in electrolyte [mol.m-3]"]

    cutoff_cases = {
        "Baseline (2.50 V)": base_cutoff,
        "-0.10 V (2.40 V)": base_cutoff - 0.10,
        "+0.10 V (2.60 V)": base_cutoff + 0.10,
    }
    c0_cases = {
        "Baseline (1000 mol m\u207b\u00b3)": base_c0,
        "-10% (900 mol m\u207b\u00b3)": base_c0 * 0.90,
        "+10% (1100 mol m\u207b\u00b3)": base_c0 * 1.10,
    }
    colors = {0: "tab:blue", 1: "tab:green", 2: "tab:red"}
    styles = {0: "-", 1: "--", 2: ":"}

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (label, v_cut) in enumerate(cutoff_cases.items()):
        dts = [run_discharge("Chen2020", c, overrides={"Lower voltage cut-off [V]": v_cut})[0]
               for c in C_RATES_SENS]
        ax_a.plot(C_RATES_SENS, dts, color=colors[i], linestyle=styles[i],
                  marker="o", markersize=5, linewidth=2.5, label=label)

    for i, (label, c0_val) in enumerate(c0_cases.items()):
        dts = [run_discharge("Chen2020", c,
                              overrides={"Initial concentration in electrolyte [mol.m-3]": c0_val})[0]
               for c in C_RATES_SENS]
        ax_b.plot(C_RATES_SENS, dts, color=colors[i], linestyle=styles[i],
                  marker="o", markersize=5, linewidth=2.5, label=label)

    for ax, title in [(ax_a, "(a) Cutoff Voltage Sensitivity"),
                       (ax_b, "(b) Initial Electrolyte Concentration Sensitivity")]:
        ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
        ax.set(xlabel="Discharge C-Rate", ylabel="Peak \u0394T (K)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cutoff Voltage and Initial Electrolyte Concentration Sensitivity\nChen2020, SPMe, 298 K",
                 fontsize=11, fontweight="bold")
    save_figure(fig, "figure15_cutoff_c0_sensitivity.png")


def generate_transport_parameter_sensitivity_figure():
    logger.info("Generating Figure 16: additional transport-parameter sensitivity...")
    p_base = pybamm.ParameterValues("Chen2020")
    transport_params = {
        "Electrolyte diffusivity": {
            "key": "Electrolyte diffusivity [m2.s-1]",
            "base_value": p_base["Electrolyte diffusivity [m2.s-1]"],
            "is_function": True,
        },
        "Electrolyte conductivity": {
            "key": "Electrolyte conductivity [S.m-1]",
            "base_value": p_base["Electrolyte conductivity [S.m-1]"],
            "is_function": True,
        },
        "Cation transference number": {
            "key": "Cation transference number",
            "base_value": p_base["Cation transference number"],
            "is_function": False,
        },
        "Separator Bruggeman coefficient": {
            "key": "Separator Bruggeman coefficient (electrolyte)",
            "base_value": p_base["Separator Bruggeman coefficient (electrolyte)"],
            "is_function": False,
        },
    }
    colors = {0: "tab:blue", 1: "tab:green", 2: "tab:red"}
    styles = {0: "-", 1: "--", 2: ":"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    for panel_idx, (pname, info) in enumerate(transport_params.items()):
        ax = axes[panel_idx]
        key = info["key"]
        for i, pct in enumerate([-0.10, 0.0, 0.10]):
            label = "Baseline" if pct == 0.0 else f"{pct:+.0%}"
            if info["is_function"]:
                override_val = scale_function_parameter(info["base_value"], 1.0 + pct)
            else:
                override_val = info["base_value"] * (1.0 + pct)
            dts = [run_discharge("Chen2020", c, overrides={key: override_val})[0] for c in C_RATES_SENS]
            logger.info("  %s, %s: dT range %.1f-%.1f K", pname, label, np.nanmin(dts), np.nanmax(dts))
            ax.plot(C_RATES_SENS, dts, color=colors[i], linestyle=styles[i],
                    marker="o", markersize=4, linewidth=2.2, label=label)
        ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
        ax.set(xlabel="Discharge C-Rate", ylabel="Peak \u0394T (K)")
        ax.set_title(f"({chr(97 + panel_idx)}) {pname} \u00b110%", fontweight="bold", fontsize=10.5)
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Additional Transport-Parameter Sensitivity (\u00b110%)\nChen2020, SPMe, 298 K",
                 fontsize=12, fontweight="bold")
    save_figure(fig, "figure16_transport_param_sensitivity.png")


def generate_bruggeman_wide_sensitivity_figure():
    logger.info("Generating Figure 17: widened Bruggeman coefficient sensitivity...")
    cases = {
        "Baseline (\u03b2 = 1.50)": 0.00,
        "-10% (\u03b2 = 1.35)": -0.10,
        "+10% (\u03b2 = 1.65)": 0.10,
        "-20% (\u03b2 = 1.20)": -0.20,
        "+20% (\u03b2 = 1.80)": 0.20,
        "-30% (\u03b2 = 1.05)": -0.30,
        "+30% (\u03b2 = 1.95)": 0.30,
    }
    wide_colors = plt.cm.coolwarm(np.linspace(0, 1, len(cases)))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    for i, (label, pct) in enumerate(cases.items()):
        b_val = BRUGGEMAN_BASE * (1.0 + pct)
        dts, durs = [], []
        for c_rate in C_RATES_SENS:
            dt, dur, _ = run_discharge(
                "Chen2020", c_rate,
                overrides={"Positive electrode Bruggeman coefficient (electrolyte)": b_val},
            )
            dts.append(dt)
            durs.append(dur)
        ls, lw = ("-", 2.2) if pct == 0 else ("--", 1.6)
        ax_a.plot(C_RATES_SENS, dts, color=wide_colors[i], linestyle=ls, linewidth=lw, marker="o", markersize=4, label=label)
        ax_b.plot(C_RATES_SENS, durs, color=wide_colors[i], linestyle=ls, linewidth=lw, marker="o", markersize=4, label=label)

    for ax, ylabel, title in [(ax_a, "Peak \u0394T (K)", "(a) Peak \u0394T vs. C-Rate"),
                               (ax_b, "Discharge Duration (s)", "(b) Discharge Duration vs. C-Rate")]:
        ax.axvline(x=2.5, color="gray", linestyle=":", alpha=0.5, label="2.5C")
        ax.set(xlabel="Discharge C-Rate", ylabel=ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7.5, ncol=1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Bruggeman Coefficient Sensitivity, Widened Range (\u00b110/20/30%)\nChen2020, SPMe, 298 K",
                 fontsize=11, fontweight="bold")
    save_figure(fig, "figure17_bruggeman_wide_sensitivity.png")


def generate_heat_decomposition_figure():
    logger.info("Generating Figure 18: heat source decomposition, SPMe vs DFN...")
    heat_vars = {
        "Ohmic": "Ohmic heating [W]",
        "Reaction": "Irreversible electrochemical heating [W]",
        "Reversible": "Reversible heating [W]",
        "Total": "Total heating [W]",
    }
    heat_colors = {"Ohmic": "tab:orange", "Reaction": "tab:purple", "Reversible": "tab:green", "Total": "black"}
    model_pairs = [("SPMe", pybamm.lithium_ion.SPMe), ("DFN", pybamm.lithium_ion.DFN)]
    c_rates = [2.25, 2.5]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col")

    for col, c_rate in enumerate(c_rates):
        for row, (model_name, model_cls) in enumerate(model_pairs):
            ax = axes[row, col]
            _, sol = simulate("Chen2020", c_rate, model_cls=model_cls, solver_mode="safe", t_points=200)
            if sol is not None:
                t = sol["Time [s]"].entries
                for label in ("Ohmic", "Reaction", "Reversible"):
                    ax.plot(t, sol[heat_vars[label]].entries, color=heat_colors[label], linewidth=2.0, label=label)
                ax.plot(t, sol[heat_vars["Total"]].entries, color=heat_colors["Total"],
                        linewidth=1.4, linestyle="--", label="Total (check)")
            ax.set_title(f"{model_name}, {c_rate}C", fontweight="bold", fontsize=10.5)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            if row == 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            if col == 0:
                ax.set_ylabel("Heat generation rate (W)", fontsize=10)

    fig.suptitle("Heat Source Decomposition vs. Time (Ohmic, Reaction, Reversible)\nChen2020, SPMe vs. DFN, 298 K",
                 fontsize=12, fontweight="bold")
    save_figure(fig, "figure18_heat_decomposition.png")


def generate_capacity_normalized_heat_figure():
    logger.info("Generating Figure 19: capacity-normalized peak heat, Chen2020 vs Ecker2015...")
    results = {"Chen2020": {"raw": [], "norm": []}, "Ecker2015": {"raw": [], "norm": []}}

    for pname in results:
        for c_rate in C_RATES_SENS:
            param, sol = simulate(pname, c_rate, t_points=150)
            nom_cap = param["Nominal cell capacity [A.h]"]
            peak_w = float(np.max(sol["Total heating [W]"].entries)) if sol is not None else np.nan
            results[pname]["raw"].append(peak_w)
            results[pname]["norm"].append(peak_w / nom_cap if not np.isnan(peak_w) else np.nan)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    for pname, color in [("Chen2020", "tab:blue"), ("Ecker2015", "tab:orange")]:
        ax_a.plot(C_RATES_SENS, results[pname]["raw"], color=color, marker="o", markersize=5, linewidth=2.2, label=pname)
        ax_b.plot(C_RATES_SENS, results[pname]["norm"], color=color, marker="o", markersize=5, linewidth=2.2, label=pname)

    ax_a.set_title("(a) Peak Total Heat Generation (Raw)", fontweight="bold")
    ax_a.set_ylabel("Peak heat generation rate (W)", fontsize=11)
    ax_b.set_title("(b) Peak Heat Generation, Normalized by Capacity", fontweight="bold")
    ax_b.set_ylabel("Peak heat generation rate per unit capacity (W A\u207b\u00b9h\u207b\u00b9)", fontsize=11)
    for ax in (ax_a, ax_b):
        ax.axvline(x=2.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Discharge C-Rate", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Capacity-Normalized Peak Heat Generation, Chen2020 vs. Ecker2015\nSPMe, 298 K",
                 fontsize=11, fontweight="bold")
    save_figure(fig, "figure19_capacity_normalized_heat.png")


def generate_ocp_swap_figure():
    logger.info("Generating Figure 20: OCP-swap counterfactual...")
    c_rate = 2.5
    prada_ocp = pybamm.ParameterValues("Prada2013")["Positive electrode OCP [V]"]

    dt_chen, dur_chen, sol_chen = run_discharge("Chen2020", c_rate, solver_mode="safe")
    dt_swap, dur_swap, sol_swap = run_discharge(
        "Chen2020", c_rate, solver_mode="safe", overrides={"Positive electrode OCP [V]": prada_ocp},
    )
    dt_prada, dur_prada, sol_prada = run_discharge("Prada2013", c_rate, solver_mode="safe")

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for sol, label, color, ls in [
        (sol_chen, f"Chen2020 native (\u0394T={dt_chen:.1f}K, t={dur_chen:.0f}s)", "tab:red", "-"),
        (sol_swap, f"Chen2020 + Prada2013 OCP (\u0394T={dt_swap:.1f}K, t={dur_swap:.0f}s)", "tab:purple", "--"),
        (sol_prada, f"Prada2013 native (\u0394T={dt_prada:.1f}K, t={dur_prada:.0f}s)", "tab:green", ":"),
    ]:
        if sol is not None:
            ax.plot(sol["Time [s]"].entries, sol["Voltage [V]"].entries, color=color, linestyle=ls,
                    linewidth=2.5, label=label)

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Terminal Voltage (V)", fontsize=11)
    ax.set_title(f"OCP-Swap Counterfactual at {c_rate}C, 298 K\nIsolating OCP Shape from Transport Parameters",
                 fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, alpha=0.3)
    save_figure(fig, "figure20_ocp_swap_counterfactual.png")


def run_dfn_convergence_check():
    """Mesh-refinement convergence check for DFN, printed (not a figure):
    default vs. refined discretization at 2.0C and 2.5C, Chen2020, 298 K."""
    logger.info("=" * 70)
    logger.info("DFN CONVERGENCE CHECK (Chen2020, 298 K)")
    logger.info("=" * 70)

    default_var_pts = pybamm.lithium_ion.DFN().default_var_pts
    keys = ["x_n", "x_s", "x_p", "r_n", "r_p"]
    refined_var_pts = dict(default_var_pts)
    refined_var_pts.update({k: 40 for k in keys})
    logger.info("Default discretization: %s", {k: default_var_pts[k] for k in keys})
    logger.info("Refined discretization: %s", {k: refined_var_pts[k] for k in keys})

    for c_rate in (2.0, 2.5):
        dt_def, dur_def, _ = run_discharge("Chen2020", c_rate, model_cls=pybamm.lithium_ion.DFN, solver_mode="safe")
        dt_ref, dur_ref, _ = run_discharge("Chen2020", c_rate, model_cls=pybamm.lithium_ion.DFN,
                                            var_pts=refined_var_pts, solver_mode="safe")
        logger.info("  %.2fC -- default: dT=%.2fK t=%.0fs | refined: dT=%.2fK t=%.0fs",
                     c_rate, dt_def, dur_def, dt_ref, dur_ref)
        if not np.isnan(dt_def) and not np.isnan(dt_ref):
            logger.info("    Difference: dT changed by %.2fK (%.1f%%), duration by %.0fs",
                         abs(dt_ref - dt_def), 100 * abs(dt_ref - dt_def) / dt_def, abs(dur_ref - dur_def))
        elif not np.isnan(dt_def):
            logger.info("    Refined-mesh run did not converge at this rate; reported as-is.")


def estimate_biot_numbers():
    """Bi = h * L_c / k, with L_c = V_cell / A_cooling and k an effective
    thickness-weighted thermal conductivity. Bi << 1 supports the lumped-
    thermal-mass assumption used throughout this study."""
    logger.info("=" * 70)
    logger.info("BIOT NUMBER ESTIMATE (all three parameter sets)")
    logger.info("=" * 70)

    h_coeff = 10.0  # W m^-2 K^-1, matches "Total heat transfer coefficient" used throughout
    for pname in ("Chen2020", "Ecker2015", "Prada2013"):
        p = load_parameter_values(pname)
        v_cell = p["Cell volume [m3]"]
        a_cell = p["Cell cooling surface area [m2]"]
        l_c = v_cell / a_cell
        try:
            k_n = p["Negative electrode thermal conductivity [W.m-1.K-1]"]
            k_s = p["Separator thermal conductivity [W.m-1.K-1]"]
            k_p = p["Positive electrode thermal conductivity [W.m-1.K-1]"]
            l_n = p["Negative electrode thickness [m]"]
            l_s = p["Separator thickness [m]"]
            l_p = p["Positive electrode thickness [m]"]
            k_eff = (k_n * l_n + k_s * l_s + k_p * l_p) / (l_n + l_s + l_p)
        except KeyError as exc:
            logger.warning("%s: missing thermal conductivity key (%s); skipping.", pname, exc)
            continue

        bi = h_coeff * l_c / k_eff
        verdict = "lumped assumption reasonable" if bi < 0.1 else "lumped assumption questionable"
        logger.info("  %s: L_c=%.3fmm, k_eff=%.2f W/m/K, Bi=%.4f (%s)", pname, l_c * 1e3, k_eff, bi, verdict)


# ============================================================================
# CLI / entry point
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate the figures for the Battery Safe Operating Envelope paper.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                         help="Directory to write figures and the run log to (default: ./figures)")
    parser.add_argument("--show", action="store_true",
                         help="Display each figure interactively as it renders (default: headless)")
    parser.add_argument("--quick", action="store_true",
                         help="Reduced-resolution smoke test: coarser grids, fewer C-rates")
    parser.add_argument("--skip-extensions", action="store_true",
                         help="Generate only the main-text figures (1-14), skipping figures 15-20 "
                              "and the DFN/Biot diagnostic checks")
    return parser.parse_args()


def configure_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "run_log.txt", mode="w"),
        ],
    )


def main():
    global OUTPUT_DIR, SHOW_PLOTS, AMBIENT_TEMPS, C_RATES, C_RATES_SENS, TEMP_LABELS, HEATMAP_KW

    args = parse_args()
    OUTPUT_DIR = args.output_dir
    SHOW_PLOTS = args.show
    configure_logging(OUTPUT_DIR)

    if args.quick:
        AMBIENT_TEMPS = np.linspace(250, 340, 3)
        C_RATES = np.array([0.5, 1.5, 2.5])
        C_RATES_SENS = np.array([0.5, 1.5, 2.5])
        TEMP_LABELS = AMBIENT_TEMPS.round(0).astype(int)
        HEATMAP_KW.update(xticklabels=C_RATES, yticklabels=TEMP_LABELS)
        logger.info("Running in --quick mode: reduced grids for a fast smoke test.")

    logger.info("PyBaMM %s", pybamm.__version__)

    generate_main_figures()
    if not args.skip_extensions:
        generate_extension_figures()

    logger.info("All figures saved successfully to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
