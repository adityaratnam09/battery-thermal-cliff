# battery-thermal-cliff
Thermal Consequences of Transport-Limited Discharge in Lithium-Ion Battery SPMe Simulations: A Safe Operating Envelope Analysis 

About the Project

When a lithium-ion battery is pushed to high discharge rates, does more current always mean more heat? This study shows that the answer is no. Using coupled Single Particle Model with electrolyte (SPMe) and lumped thermal simulations in PyBaMM, we map how peak temperature rise (ΔT) changes across a 10×9 grid of ambient temperatures (250–340 K) and discharge rates (0.5–2.5C). For an NMC INR21700 M50 cell (Chen2020 [8]), ΔT rises steadily with C-rate up to 2.25C, reaching 64.9 K, then drops sharply at 2.5C — a “thermal cliff”. The mechanism is Li⁺ depletion at the cathode surface: at 2.5C, diffusion-limited lithium-ion transport in the electrolyte triggers premature voltage cutoff after just 10% of the theoretical discharge duration, sharply reducing total heat generation. Spatial concentration profiles, a Bruggeman sensitivity analysis (±10%), and DFN model comparison provide converging mechanistic evidence: the cliff persists across parameter perturbations, but at 2.5C the DFN runs to ∼72% capacity versus ∼10% in SPMe, showing the cliff location is sensitive to electrolyte model fidelity. LFP cells (Ecker2015) show negligible ΔT (0–4 K) and no early cutoff, consistent with a mechanistically distinct discharge regime. These findings suggest that BMS algorithms must account for electrolyte-side transport constraints that can abruptly alter the thermal trajectory of the cell.

Data Generation Steps

1. Ensure PyBaMM is installed.
2. Run the simulation script: python battery_safe_operating_envelope.py.
