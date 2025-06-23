"""
Author: danikae
Date: 250623
Purpose: Makes a bar plot to compare conversion costs across different vessels
"""

import matplotlib.pyplot as plt
from parse import parse
from collections import defaultdict
from common_tools import get_top_dir

# --- Vessel and fuel metadata ---

vessel_size_title_with_class = {
    "bulk_carrier_capesize_ice": "Bulk Carrier (Capesize)",
    "bulk_carrier_handy_ice": "Bulk Carrier (Handy)",
    "bulk_carrier_panamax_ice": "Bulk Carrier (Panamax)",
    "container_15000_teu_ice": "Container (15,000 TEU)",
    "container_8000_teu_ice": "Container (8,000 TEU)",
    "container_3500_teu_ice": "Container (3,500 TEU)",
    "tanker_100k_dwt_ice": "Tanker (100k DWT)",
    "tanker_300k_dwt_ice": "Tanker (300k DWT)",
    "tanker_35k_dwt_ice": "Tanker (35k DWT)",
    "gas_carrier_100k_cbm_ice": "Gas Carrier (100k m$^3$)",
}

fuel_colors = {
    "ammonia": "tab:blue",
    "hydrogen": "tab:orange",
    "liquid_hydrogen": "tab:orange",
    "methanol": "tab:green",
    "diesel": "tab:red",
    "oil": "black",
    "lsfo": "black",
    "lng": "tab:grey",
    "methane": "tab:grey"
}

vessel_names = {
    "ammonia": "Ammonia (dual fuel)",
    "liquid_hydrogen": "Liquid Hydrogen (dual fuel)",
    "methanol": "Methanol (dual fuel)",
    "methane": "Methane (dual fuel)",
    "hydrogen": "Liquid Hydrogen (dual fuel)",
    "oil": "LSFO (single fuel)",
    "lsfo": "LSFO (single fuel)",
    "diesel": "FT Diesel (single fuel)"
}

default_hatches = ["", "xxx", "+++", "***", "///", "\\\\\\", "ooo", "..."]

# --- Parse fuel conversion cost file ---

conversion_data = []
pattern = 'set_fuel_conversion_cost("{src}", "{tgt}", {cost:g})'

top_dir = get_top_dir()
with open(f"{top_dir}/includes_global/fuel_conversion_costs_multi_fleet.inc", "r") as f:
    for line in f:
        line = line.strip().split("#")[0].strip()  # Remove inline comment if present
        if not line:
            continue
        result = parse(pattern, line)
        if result:
            src = result['src']
            tgt = result['tgt']
            cost = result['cost']
            vessel_key_ice = "_".join(src.split("_")[:-1])
            from_fuel = src.split("_")[-1]
            to_fuel = tgt.split("_")[-1]
            conversion_data.append({
                "vessel": vessel_key_ice,
                "from_fuel": from_fuel,
                "to_fuel": to_fuel,
                "cost": cost / 1e6
            })

# --- Organize data ---

valid_vessels = [v for v in vessel_size_title_with_class if any(d["vessel"] == v for d in conversion_data)]
conversion_pairs = sorted(set((d["from_fuel"], d["to_fuel"]) for d in conversion_data))
target_fuels = sorted(set(to for _, to in conversion_pairs))

# Detect which target fuels have multiple source fuels (need hatch)
targets_with_multiple_sources = {to_fuel for to_fuel in target_fuels
                                 if len({from_fuel for from_fuel, t in conversion_pairs if t == to_fuel}) > 1}

# Assign hatches only where needed
hatch_map = {}
hatch_index = 0
for from_fuel, to_fuel in conversion_pairs:
    if to_fuel in targets_with_multiple_sources:
        hatch_map[(from_fuel, to_fuel)] = default_hatches[hatch_index % len(default_hatches)]
        hatch_index += 1
    else:
        hatch_map[(from_fuel, to_fuel)] = ""  # no hatch needed

# --- Plotting ---

bar_width = 0.15
group_spacing = 0.5
fig, ax = plt.subplots(figsize=(14, 7))

yticks, yticklabels = [], []
bar_positions, bar_values = [], []
bar_colors, bar_labels, bar_hatches = [], [], []

y = 0
for vessel in valid_vessels:
    vessel_data = [d for d in conversion_data if d["vessel"] == vessel]
    pair_to_cost = {(d["from_fuel"], d["to_fuel"]): d["cost"] for d in vessel_data}
    for i, (from_fuel, to_fuel) in enumerate(conversion_pairs):
        if (from_fuel, to_fuel) in pair_to_cost:
            xpos = y + i * bar_width
            bar_positions.append(xpos)
            bar_values.append(pair_to_cost[(from_fuel, to_fuel)])
            bar_colors.append(fuel_colors.get(to_fuel, "gray"))
            bar_labels.append((from_fuel, to_fuel))
            bar_hatches.append(hatch_map.get((from_fuel, to_fuel), ""))
    yticks.append(y + (len(conversion_pairs) - 1) * bar_width / 2)
    yticklabels.append(vessel_size_title_with_class[vessel])
    y += len(conversion_pairs) * bar_width + group_spacing

# --- Draw bars ---

bars = []
for xpos, val, color, hatch in zip(bar_positions, bar_values, bar_colors, bar_hatches):
    b = ax.barh(xpos, val, height=bar_width, color=color, hatch=hatch, edgecolor="black")
    bars.append(b[0])

# --- Legend (distinct label per unique (from, to) pair) ---

legend_entries = {}
for bar, (from_fuel, to_fuel), hatch in zip(bars, bar_labels, bar_hatches):
    label = f"{vessel_names.get(from_fuel.lower(), from_fuel)}\n→ {vessel_names.get(to_fuel.lower(), to_fuel)}"
    key = (label, hatch)
    if key not in legend_entries:
        legend_entries[key] = plt.Rectangle((0, 0), 1, 1,
                                            facecolor=bar.get_facecolor(),
                                            hatch=hatch,
                                            edgecolor="black")

ax.legend(
    legend_entries.values(),
    [label for (label, _) in legend_entries.keys()],
    title="Fuel Conversion",
    fontsize=14,
    title_fontsize=16,
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

# --- Final layout ---

ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True)
ax.set_xlabel("Conversion Cost (Million USD)", fontsize=18)
#ax.set_title("Fuel Conversion Costs by Vessel and Target Fuel")
plt.tight_layout()

print(f"Saving to plots/conversion_costs.png")
plt.savefig("plots/conversion_costs.png", dpi=300)
plt.savefig("plots/conversion_costs.pdf")
