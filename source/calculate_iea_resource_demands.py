"""
Date: 251120
Purpose: Calculate resource demands associated with the IEA net zero 2050 scenario for maritime shipping
"""

import pandas as pd
from common_tools import get_fuel_LHV

KG_PER_TONNE = 1000
TONNES_PER_MEGATONNE = 1e6

# Scenario inputs
total_energy_consumed = 10      # EJ
ammonia_energy_frac = 0.46
hydrogen_energy_frac = 0.17
biofuel_energy_frac = 0.21

# Stoichiometric hydrogen requirement for ammonia
kg_hydrogen_per_kg_ammonia = 0.17756899588960657663  # kg H2 / kg NH3

# Input data files
filenames = {
    "hydrogen": "input_fuel_pathway_data/production/hydrogen_resource_demands.csv",
    "ammonia": "input_fuel_pathway_data/production/ammonia_resource_demands.csv",
    "liquid_hydrogen": "input_fuel_pathway_data/production/liquid_hydrogen_resource_demands.csv",
    "biofuel": "input_fuel_pathway_data/production/bio_cfp_resource_demands.csv",
    "hydrogen_to_ammonia": "input_fuel_pathway_data/process/hydrogen_to_ammonia_conversion_resource_demands.csv",
    "hydrogen_liquefaction": "input_fuel_pathway_data/process/hydrogen_liquefaction_resource_demands.csv"
}

# Names used to look up LHVs via get_fuel_LHV
fuel_name_for_lhv = {
    "ammonia": "ammonia",
    "liquid_hydrogen": "liquid_hydrogen",
    "biofuel": "bio_cfp"
}

h_sources = ["LTE", "ATRCCS", "SMR", "BG"]

# Column names used everywhere
RESOURCE_COLS = [
    "Electricity Demand [kWh / kg fuel]",
    "Lignocellulosic Biomass Demand [kg / kg fuel]",
    "NG Demand [GJ / kg fuel]",
    "Water Demand [m^3 / kg fuel]"
]


def get_resources(filename, h_source=None):
    """
    Read resource demands per kg of 'fuel' from a CSV.
    If h_source is provided, filter by it and (if multiple rows) average.
    If h_source is None, require that there is only one row.
    """
    df = pd.read_csv(filename)

    if h_source is None:
        # Only one row allowed if no hydrogen source is specified
        if len(df) == 1:
            return df[RESOURCE_COLS].iloc[0]
        else:
            raise ValueError(
                f"Multiple rows in {filename} — please specify h_source explicitly."
            )
    else:
        filtered = df[df["Hydrogen Source"] == h_source]

        if filtered.empty:
            raise ValueError(f"No rows found for hydrogen source '{h_source}' in {filename}")

        # Average if multiple rows (e.g., grid/solar/wind variants)
        return filtered[RESOURCE_COLS].mean(numeric_only=True)


def get_process_resources(filename):
    """
    For process-level files (hydrogen_to_ammonia, hydrogen_liquefaction),
    resource demands are treated as independent of hydrogen source and
    averaged across all rows.
    """
    df = pd.read_csv(filename)
    return df[RESOURCE_COLS].mean(numeric_only=True)


def build_hydrogen_perkg_by_source():
    """
    Build a dict mapping each h_source to the per-kg hydrogen production
    resource demands.
    """
    hydrogen_perkg_by_source = {}

    for h in h_sources:
        try:
            res = get_resources(filenames["hydrogen"], h_source=h)
        except ValueError:
            # Skip if no hydrogen pathway for this source
            continue

        # res is already a Series with RESOURCE_COLS keys
        hydrogen_perkg_by_source[h] = res

    return hydrogen_perkg_by_source


def get_resources_by_fuel(hydrogen_perkg_by_source,
                          h2_to_nh3_perkg,
                          h2_liq_perkg):
    """
    Build per-kg resource demands for FINAL fuels (ammonia, liquid hydrogen, biofuel).

    For ammonia and liquid hydrogen, we assume the corresponding production CSVs
    already represent the total production process, so we use them directly.

    Returns a DataFrame with columns:
      Fuel, Hydrogen Source, and per-kg resource columns.
    """
    records = []

    # ammonia
    for h in h_sources:
        try:
            res_ammonia = get_resources(filenames["ammonia"], h_source=h)
        except ValueError:
            continue

        total_perkg_ammonia = res_ammonia  # use directly

        record = {
            "Fuel": "ammonia",
            "Hydrogen Source": h,
        }
        record.update(total_perkg_ammonia.to_dict())
        records.append(record)

    # liquid hydrogen
    for h in h_sources:
        try:
            res_lh2 = get_resources(filenames["liquid_hydrogen"], h_source=h)
        except ValueError:
            continue

        total_perkg_lh2 = res_lh2  # use directly; no extra additions

        record = {
            "Fuel": "liquid_hydrogen",
            "Hydrogen Source": h,
        }
        record.update(total_perkg_lh2.to_dict())
        records.append(record)

    # biofuel (no hydrogen in this chain)
    bio_res = get_resources(filenames["biofuel"])
    record = {
        "Fuel": "biofuel",
        "Hydrogen Source": "n/a",
    }
    record.update(bio_res.to_dict())
    records.append(record)

    return pd.DataFrame(records)


def calculate_resource_totals(resources_df: pd.DataFrame) -> pd.DataFrame:
    """
    Using resource demands per kg of FINAL fuel, fuel energy fractions, and LHVs,
    calculate the total resource demand in the 10 EJ IEA NZ2050 scenario
    for each (fuel, hydrogen source) combination.
    """
    # Map final fuel to its energy fraction in the shipping scenario
    energy_fracs = {
        "ammonia": ammonia_energy_frac,
        "liquid_hydrogen": hydrogen_energy_frac,
        "biofuel": biofuel_energy_frac,
    }

    records = []

    for _, row in resources_df.iterrows():
        fuel = row["Fuel"]
        h_source = row["Hydrogen Source"]

        # Skip any fuels not in our energy fraction mapping
        if fuel not in energy_fracs:
            continue

        energy_frac = energy_fracs[fuel]  # fraction of total 10 EJ
        fuel_energy_EJ = total_energy_consumed * energy_frac  # EJ

        # Lower heating value in MJ/kg
        lhv_MJ_per_kg = get_fuel_LHV(fuel_name_for_lhv[fuel])

        # 1 EJ = 1e18 J = 1e12 MJ
        fuel_mass_kg = fuel_energy_EJ * 1e12 / lhv_MJ_per_kg

        # Resource demands per kg of this fuel:
        elec_per_kg = row["Electricity Demand [kWh / kg fuel]"]
        lcb_per_kg = row["Lignocellulosic Biomass Demand [kg / kg fuel]"]
        ng_per_kg = row["NG Demand [GJ / kg fuel]"]
        water_per_kg = row["Water Demand [m^3 / kg fuel]"]

        # Total resource demands for the scenario
        total_elec_kWh = elec_per_kg * fuel_mass_kg
        total_lcb_kg = lcb_per_kg * fuel_mass_kg
        total_ng_GJ = ng_per_kg * fuel_mass_kg
        total_water_cbm = water_per_kg * fuel_mass_kg

        records.append(
            {
                "Fuel": fuel,
                "Hydrogen Source": h_source,
                "Fuel Energy Fraction": energy_frac,
                "Fuel Energy in Scenario [EJ]": fuel_energy_EJ,
                "Fuel LHV [MJ/kg]": lhv_MJ_per_kg,
                "Fuel Mass in Scenario [Mt]": fuel_mass_kg / (KG_PER_TONNE * TONNES_PER_MEGATONNE),
                "Electricity Demand [kWh]": total_elec_kWh,
                "LCB Demand [kg]": total_lcb_kg,
                "NG Demand [GJ]": total_ng_GJ,
                "Water Demand [m^3]": total_water_cbm
            }
        )

    return pd.DataFrame(records)


def add_explicit_hydrogen_rows(totals_df: pd.DataFrame,
                               hydrogen_perkg_by_source,
                               h2_to_nh3_perkg,
                               h2_liq_perkg) -> pd.DataFrame:
    """
    Add explicit rows for:
      - hydrogen_for_ammonia        (include mass & energy of hydrogen)
      - hydrogen_to_ammonia_conversion   (NO mass/energy)
      - hydrogen_for_lh2            (include mass & energy of hydrogen)
      - hydrogen_liquefaction       (NO mass/energy)
    """
    records = []

    for h in h_sources:

        # ------------------------------------------------------------------
        # Parent AMMONIA row for this source
        # ------------------------------------------------------------------
        nh3_rows = totals_df[
            (totals_df["Fuel"] == "ammonia") &
            (totals_df["Hydrogen Source"] == h)
        ]
        nh3_row = nh3_rows.iloc[0] if len(nh3_rows) == 1 else None

        if nh3_row is not None and h in hydrogen_perkg_by_source:

            h2_prod = hydrogen_perkg_by_source[h]

            # Parent ammonia mass and energy
            nh3_energy_EJ = nh3_row["Fuel Energy in Scenario [EJ]"]
            nh3_mass_Mt   = nh3_row["Fuel Mass in Scenario [Mt]"]
            nh3_mass_kg   = nh3_mass_Mt * 1e9

            # Hydrogen needs (mass & energy)
            h2_mass_kg = nh3_mass_kg * kg_hydrogen_per_kg_ammonia
            h2_mass_Mt = h2_mass_kg / 1e9
            h2_energy_EJ = nh3_energy_EJ * kg_hydrogen_per_kg_ammonia

            # --- hydrogen_for_ammonia ---
            r = h2_prod * h2_mass_kg
            records.append({
                "Fuel": "hydrogen_for_ammonia",
                "Hydrogen Source": h,
                "Fuel Energy Fraction": None,
                "Fuel Energy in Scenario [EJ]": h2_energy_EJ,
                "Fuel LHV [MJ/kg]": None,
                "Fuel Mass in Scenario [Mt]": h2_mass_Mt,
                "Electricity Demand [kWh]": r["Electricity Demand [kWh / kg fuel]"],
                "LCB Demand [kg]": r["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                "NG Demand [GJ]": r["NG Demand [GJ / kg fuel]"],
                "Water Demand [m^3]": r["Water Demand [m^3 / kg fuel]"]
            })

            # --- hydrogen_to_ammonia_conversion (NO mass/energy) ---
            conv = h2_to_nh3_perkg * nh3_mass_kg
            records.append({
                "Fuel": "hydrogen_to_ammonia_conversion",
                "Hydrogen Source": h,
                "Fuel Energy Fraction": None,
                "Fuel Energy in Scenario [EJ]": None,
                "Fuel LHV [MJ/kg]": None,
                "Fuel Mass in Scenario [Mt]": None,
                "Electricity Demand [kWh]": conv["Electricity Demand [kWh / kg fuel]"],
                "LCB Demand [kg]": conv["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                "NG Demand [GJ]": conv["NG Demand [GJ / kg fuel]"],
                "Water Demand [m^3]": conv["Water Demand [m^3 / kg fuel]"]
            })

        # ------------------------------------------------------------------
        # Parent LIQUID HYDROGEN row for this source
        # ------------------------------------------------------------------
        lh2_rows = totals_df[
            (totals_df["Fuel"] == "liquid_hydrogen") &
            (totals_df["Hydrogen Source"] == h)
        ]
        lh2_row = lh2_rows.iloc[0] if len(lh2_rows) == 1 else None

        if lh2_row is not None and h in hydrogen_perkg_by_source:

            h2_prod = hydrogen_perkg_by_source[h]

            # Parent LH2 mass & energy
            lh2_energy_EJ = lh2_row["Fuel Energy in Scenario [EJ]"]
            lh2_mass_Mt   = lh2_row["Fuel Mass in Scenario [Mt]"]
            lh2_mass_kg   = lh2_mass_Mt * 1e9

            # Hydrogen input = 1 kg H2 / 1 kg LH2
            h2_mass_kg = lh2_mass_kg
            h2_mass_Mt = lh2_mass_Mt
            h2_energy_EJ = lh2_energy_EJ

            # --- hydrogen_for_lh2 ---
            r = h2_prod * h2_mass_kg
            records.append({
                "Fuel": "hydrogen_for_lh2",
                "Hydrogen Source": h,
                "Fuel Energy Fraction": None,
                "Fuel Energy in Scenario [EJ]": h2_energy_EJ,
                "Fuel LHV [MJ/kg]": None,
                "Fuel Mass in Scenario [Mt]": h2_mass_Mt,
                "Electricity Demand [kWh]": r["Electricity Demand [kWh / kg fuel]"],
                "LCB Demand [kg]": r["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                "NG Demand [GJ]": r["NG Demand [GJ / kg fuel]"],
                "Water Demand [m^3]": r["Water Demand [m^3 / kg fuel]"]
            })

            # --- hydrogen_liquefaction (NO mass/energy) ---
            liq = h2_liq_perkg * lh2_mass_kg
            records.append({
                "Fuel": "hydrogen_liquefaction",
                "Hydrogen Source": h,
                "Fuel Energy Fraction": None,
                "Fuel Energy in Scenario [EJ]": None,
                "Fuel LHV [MJ/kg]": None,
                "Fuel Mass in Scenario [Mt]": None,
                "Electricity Demand [kWh]": liq["Electricity Demand [kWh / kg fuel]"],
                "LCB Demand [kg]": liq["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                "NG Demand [GJ]": liq["NG Demand [GJ / kg fuel]"],
                "Water Demand [m^3]": liq["Water Demand [m^3 / kg fuel]"]
            })

    return pd.DataFrame(records)


def make_pathway_tables(full_output: pd.DataFrame):
    """
    For each hydrogen pathway (hydrogen source), create a CSV with rows:
      - liquid_hydrogen, hydrogen_for_lh2
      - liquid_hydrogen, hydrogen_liquefaction
      - liquid_hydrogen, liquid_hydrogen_final
      - ammonia, hydrogen_for_ammonia
      - ammonia, hydrogen_to_ammonia_conversion
      - ammonia, ammonia_final
      - biofuel, biofuel_final

    Columns:
      - Fuel being produced
      - Process or final fuel
      - Fuel energy [EJ]
      - Fuel mass [Mt]
      - Electricity Demand [kWh]
      - LCB Demand [kg]
      - NG Demand [GJ]
      - Water Demand [m^3]
    """
    # Biofuel total row is independent of H source
    bio_row = full_output[full_output["Fuel"] == "biofuel"].iloc[0]

    for h in h_sources:
        rows = []

        # Helper to safely extract a single row by (Fuel, Hydrogen Source)
        def get_row(fuel_value, h_source_value):
            df = full_output[
                (full_output["Fuel"] == fuel_value) &
                (full_output["Hydrogen Source"] == h_source_value)
            ]
            return df.iloc[0] if len(df) == 1 else None

        # liquid_hydrogen, hydrogen_for_lh2
        r = get_row("hydrogen_for_lh2", h)
        if r is not None:
            rows.append({
                "Fuel being produced": "liquid_hydrogen",
                "Process or final fuel": "hydrogen_for_lh2",
                "Fuel energy [EJ]": r["Fuel Energy in Scenario [EJ]"],
                "Fuel mass [Mt]": r["Fuel Mass in Scenario [Mt]"],
                "Electricity Demand [kWh]": r["Electricity Demand [kWh]"],
                "LCB Demand [kg]": r["LCB Demand [kg]"],
                "NG Demand [GJ]": r["NG Demand [GJ]"],
                "Water Demand [m^3]": r["Water Demand [m^3]"],
            })

        # liquid_hydrogen, hydrogen_liquefaction
        r = get_row("hydrogen_liquefaction", h)
        if r is not None:
            rows.append({
                "Fuel being produced": "liquid_hydrogen",
                "Process or final fuel": "hydrogen_liquefaction",
                "Fuel energy [EJ]": None,
                "Fuel mass [Mt]": None,
                "Electricity Demand [kWh]": r["Electricity Demand [kWh]"],
                "LCB Demand [kg]": r["LCB Demand [kg]"],
                "NG Demand [GJ]": r["NG Demand [GJ]"],
                "Water Demand [m^3]": r["Water Demand [m^3]"],
            })

        # liquid_hydrogen, liquid_hydrogen_final
        r = get_row("liquid_hydrogen", h)
        if r is not None:
            rows.append({
                "Fuel being produced": "liquid_hydrogen",
                "Process or final fuel": "liquid_hydrogen_final",
                "Fuel energy [EJ]": r["Fuel Energy in Scenario [EJ]"],
                "Fuel mass [Mt]": r["Fuel Mass in Scenario [Mt]"],
                "Electricity Demand [kWh]": r["Electricity Demand [kWh]"],
                "LCB Demand [kg]": r["LCB Demand [kg]"],
                "NG Demand [GJ]": r["NG Demand [GJ]"],
                "Water Demand [m^3]": r["Water Demand [m^3]"],
            })

        # ammonia, hydrogen_for_ammonia
        r = get_row("hydrogen_for_ammonia", h)
        if r is not None:
            rows.append({
                "Fuel being produced": "ammonia",
                "Process or final fuel": "hydrogen_for_ammonia",
                "Fuel energy [EJ]": r["Fuel Energy in Scenario [EJ]"],
                "Fuel mass [Mt]": r["Fuel Mass in Scenario [Mt]"],
                "Electricity Demand [kWh]": r["Electricity Demand [kWh]"],
                "LCB Demand [kg]": r["LCB Demand [kg]"],
                "NG Demand [GJ]": r["NG Demand [GJ]"],
                "Water Demand [m^3]": r["Water Demand [m^3]"],
            })

        # ammonia, hydrogen_to_ammonia_conversion
        r = get_row("hydrogen_to_ammonia_conversion", h)
        if r is not None:
            rows.append({
                "Fuel being produced": "ammonia",
                "Process or final fuel": "hydrogen_to_ammonia_conversion",
                "Fuel energy [EJ]": None,
                "Fuel mass [Mt]": None,
                "Electricity Demand [kWh]": r["Electricity Demand [kWh]"],
                "LCB Demand [kg]": r["LCB Demand [kg]"],
                "NG Demand [GJ]": r["NG Demand [GJ]"],
                "Water Demand [m^3]": r["Water Demand [m^3]"],
            })

        # ammonia, ammonia_final
        r = get_row("ammonia", h)
        if r is not None:
            rows.append({
                "Fuel being produced": "ammonia",
                "Process or final fuel": "ammonia_final",
                "Fuel energy [EJ]": r["Fuel Energy in Scenario [EJ]"],
                "Fuel mass [Mt]": r["Fuel Mass in Scenario [Mt]"],
                "Electricity Demand [kWh]": r["Electricity Demand [kWh]"],
                "LCB Demand [kg]": r["LCB Demand [kg]"],
                "NG Demand [GJ]": r["NG Demand [GJ]"],
                "Water Demand [m^3]": r["Water Demand [m^3]"],
            })

        # biofuel, biofuel_final (same entry repeated in each file)
        rows.append({
            "Fuel being produced": "biofuel",
            "Process or final fuel": "biofuel_final",
            "Fuel energy [EJ]": bio_row["Fuel Energy in Scenario [EJ]"],
            "Fuel mass [Mt]": bio_row["Fuel Mass in Scenario [Mt]"],
            "Electricity Demand [kWh]": bio_row["Electricity Demand [kWh]"],
            "LCB Demand [kg]": bio_row["LCB Demand [kg]"],
            "NG Demand [GJ]": bio_row["NG Demand [GJ]"],
            "Water Demand [m^3]": bio_row["Water Demand [m^3]"],
        })

        out_df = pd.DataFrame(rows)

        # Save one file per hydrogen pathway
        out_path = f"tables/resource_demands_pathway_{h}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved pathway table for {h} to {out_path}")


def main():
    # 1) Build hydrogen per-kg resource demands by source
    hydrogen_perkg_by_source = build_hydrogen_perkg_by_source()

    # 2) Process-level per-kg resource demands
    h2_to_nh3_perkg = get_process_resources(filenames["hydrogen_to_ammonia"])
    h2_liq_perkg = get_process_resources(filenames["hydrogen_liquefaction"])

    # 3) Per-kg resource demands for FINAL fuels
    resources_by_fuel = get_resources_by_fuel(
        hydrogen_perkg_by_source,
        h2_to_nh3_perkg,
        h2_liq_perkg
    )
    print("Per-kg resource demands by fuel / hydrogen source:")
    print(resources_by_fuel)

    # 4) Scenario-total resource demands by final fuel
    totals_df = calculate_resource_totals(resources_by_fuel)
    print("\nTotal resource demands in 10 EJ NZ2050 scenario (by final fuel):")
    print(totals_df)

    # 5) Explicit hydrogen-related component rows
    explicit_h_rows = add_explicit_hydrogen_rows(
        totals_df,
        hydrogen_perkg_by_source,
        h2_to_nh3_perkg,
        h2_liq_perkg
    )

    # 6) Combine everything into one output table
    full_output = pd.concat([totals_df, explicit_h_rows], ignore_index=True)

    full_output.to_csv(
        "tables/total_resource_demands_iea_2050_with_h2_components.csv",
        index=False
    )

    print("\nFull output including explicit hydrogen component rows:")
    print(full_output)

    # 7) Build per-pathway summary files
    make_pathway_tables(full_output)


if __name__ == "__main__":
    main()
