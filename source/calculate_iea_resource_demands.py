"""
Date: 251120
Purpose: Calculate resource demands associated with the IEA net zero 2050 scenario for maritime shipping
"""

import pandas as pd
from common_tools import get_fuel_LHV

# Scenario inputs
total_energy_consumed = 10      # EJ
ammonia_energy_frac = 0.46
hydrogen_energy_frac = 0.17
biofuel_energy_frac = 0.21

# Stoichiometric hydrogen requirement for ammonia
kg_hydrogen_per_kg_ammonia = 0.355599884  # kg H2 / kg NH3

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
    Build per-kg resource demands for FINAL fuels (ammonia, liquid hydrogen, biofuel),
    including the upstream resource demands to:
      - produce hydrogen used for ammonia and liquid H2, and
      - convert hydrogen to ammonia, and liquefy hydrogen.

    Returns a DataFrame with columns:
      Fuel, Hydrogen Source, and per-kg resource columns.
    """
    records = []

    # ---- Final fuel: ammonia (includes NH3 production + H2 prod + H2->NH3 conversion)
    for h in h_sources:
        # ammonia production (per kg NH3) for this source
        try:
            res_ammonia = get_resources(filenames["ammonia"], h_source=h)
        except ValueError:
            # no ammonia pathway for this h_source
            continue

        # hydrogen production per kg H2 for this source
        if h not in hydrogen_perkg_by_source:
            continue

        h2_prod_perkg = hydrogen_perkg_by_source[h]

        # Per kg NH3:
        #   ammonia production
        # + hydrogen production (scaled by stoichiometric H2 per kg NH3)
        # + hydrogen-to-ammonia conversion (per kg NH3)
        total_perkg_ammonia = (
            res_ammonia
            + h2_prod_perkg * kg_hydrogen_per_kg_ammonia
            + h2_to_nh3_perkg
        )

        record = {
            "Fuel": "ammonia",
            "Hydrogen Source": h,
        }
        record.update(total_perkg_ammonia.to_dict())
        records.append(record)

    # ---- Final fuel: liquid hydrogen (includes LH2 prod + H2 prod + liquefaction)
    # Assume 1 kg H2 per kg liquid H2, with no H2 loss.
    for h in h_sources:
        try:
            res_lh2 = get_resources(filenames["liquid_hydrogen"], h_source=h)
        except ValueError:
            continue

        if h not in hydrogen_perkg_by_source:
            continue

        h2_prod_perkg = hydrogen_perkg_by_source[h]

        total_perkg_lh2 = (
            res_lh2
            + h2_prod_perkg * 1.0   # 1 kg H2 per kg LH2 (no loss)
            + h2_liq_perkg
        )

        record = {
            "Fuel": "liquid_hydrogen",
            "Hydrogen Source": h,
        }
        record.update(total_perkg_lh2.to_dict())
        records.append(record)

    # ---- Final fuel: biofuel (no hydrogen in this chain)
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
    Using resource demands per kg of FINAL fuel (which already include upstream hydrogen
    and process steps where relevant), fuel energy fractions, and LHVs,
    calculate the total resource demand in the 10 EJ IEA NZ2050 scenario
    for each (fuel, hydrogen source) combination.

    Returns a DataFrame with total electricity, LCB, NG, and water demands.
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
        # mass of fuel (kg) needed for this fuel's share of the scenario:
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
                "Total Fuel Mass in Scenario [kg]": fuel_mass_kg,
                "Total Electricity Demand [kWh]": total_elec_kWh,
                "Total LCB Demand [kg]": total_lcb_kg,
                "Total NG Demand [GJ]": total_ng_GJ,
                "Total Water Demand [m^3]": total_water_cbm
            }
        )

    return pd.DataFrame(records)


def add_explicit_hydrogen_rows(totals_df: pd.DataFrame,
                               hydrogen_perkg_by_source,
                               h2_to_nh3_perkg,
                               h2_liq_perkg) -> pd.DataFrame:
    """
    Add explicit rows (scenario totals) for each h_source:

      - hydrogen_for_ammonia
      - hydrogen_to_ammonia_conversion
      - hydrogen_for_lh2
      - hydrogen_liquefaction

    These rows decompose the total hydrogen-related demands used to support
    ammonia and liquid hydrogen in the scenario.
    """
    records = []

    for h in h_sources:
        # ------------------------------------------------------------------
        # Hydrogen for ammonia
        # ------------------------------------------------------------------
        if h in hydrogen_perkg_by_source:
            h2_prod = hydrogen_perkg_by_source[h]

            nh3_mass_arr = totals_df[
                (totals_df["Fuel"] == "ammonia") &
                (totals_df["Hydrogen Source"] == h)
            ]["Total Fuel Mass in Scenario [kg]"].values

            if len(nh3_mass_arr) == 1:
                nh3_mass = nh3_mass_arr[0]
                hydrogen_mass_needed = nh3_mass * kg_hydrogen_per_kg_ammonia

                # Scenario-total hydrogen production resources for ammonia
                r = h2_prod * hydrogen_mass_needed
                records.append({
                    "Fuel": "hydrogen_for_ammonia",
                    "Hydrogen Source": h,
                    "Fuel Energy Fraction": None,
                    "Fuel Energy in Scenario [EJ]": None,
                    "Fuel LHV [MJ/kg]": None,
                    "Total Fuel Mass in Scenario [kg]": hydrogen_mass_needed,
                    "Total Electricity Demand [kWh]": r["Electricity Demand [kWh / kg fuel]"],
                    "Total LCB Demand [kg]": r["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                    "Total NG Demand [GJ]": r["NG Demand [GJ / kg fuel]"],
                    "Total Water Demand [m^3]": r["Water Demand [m^3 / kg fuel]"]
                })

                # ------------------------------------------------------------------
                # Hydrogen-to-ammonia conversion (per kg NH3, scaled to scenario)
                # ------------------------------------------------------------------
                conv = h2_to_nh3_perkg * nh3_mass
                records.append({
                    "Fuel": "hydrogen_to_ammonia_conversion",
                    "Hydrogen Source": h,
                    "Fuel Energy Fraction": None,
                    "Fuel Energy in Scenario [EJ]": None,
                    "Fuel LHV [MJ/kg]": None,
                    "Total Fuel Mass in Scenario [kg]": nh3_mass,
                    "Total Electricity Demand [kWh]": conv["Electricity Demand [kWh / kg fuel]"],
                    "Total LCB Demand [kg]": conv["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                    "Total NG Demand [GJ]": conv["NG Demand [GJ / kg fuel]"],
                    "Total Water Demand [m^3]": conv["Water Demand [m^3 / kg fuel]"]
                })

        # ------------------------------------------------------------------
        # Hydrogen for liquid hydrogen
        # ------------------------------------------------------------------
        if h in hydrogen_perkg_by_source:
            h2_prod = hydrogen_perkg_by_source[h]

            lh2_mass_arr = totals_df[
                (totals_df["Fuel"] == "liquid_hydrogen") &
                (totals_df["Hydrogen Source"] == h)
            ]["Total Fuel Mass in Scenario [kg]"].values

            if len(lh2_mass_arr) == 1:
                lh2_mass = lh2_mass_arr[0]
                hydrogen_mass_needed = lh2_mass  # 1 kg H2 per kg LH2, no loss

                r = h2_prod * hydrogen_mass_needed
                records.append({
                    "Fuel": "hydrogen_for_lh2",
                    "Hydrogen Source": h,
                    "Fuel Energy Fraction": None,
                    "Fuel Energy in Scenario [EJ]": None,
                    "Fuel LHV [MJ/kg]": None,
                    "Total Fuel Mass in Scenario [kg]": hydrogen_mass_needed,
                    "Total Electricity Demand [kWh]": r["Electricity Demand [kWh / kg fuel]"],
                    "Total LCB Demand [kg]": r["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                    "Total NG Demand [GJ]": r["NG Demand [GJ / kg fuel]"],
                    "Total Water Demand [m^3]": r["Water Demand [m^3 / kg fuel]"]
                })

                # ------------------------------------------------------------------
                # Hydrogen liquefaction (per kg LH2, scaled to scenario)
                # ------------------------------------------------------------------
                liq = h2_liq_perkg * lh2_mass
                records.append({
                    "Fuel": "hydrogen_liquefaction",
                    "Hydrogen Source": h,
                    "Fuel Energy Fraction": None,
                    "Fuel Energy in Scenario [EJ]": None,
                    "Fuel LHV [MJ/kg]": None,
                    "Total Fuel Mass in Scenario [kg]": lh2_mass,
                    "Total Electricity Demand [kWh]": liq["Electricity Demand [kWh / kg fuel]"],
                    "Total LCB Demand [kg]": liq["Lignocellulosic Biomass Demand [kg / kg fuel]"],
                    "Total NG Demand [GJ]": liq["NG Demand [GJ / kg fuel]"],
                    "Total Water Demand [m^3]": liq["Water Demand [m^3 / kg fuel]"]
                })

    return pd.DataFrame(records)


def main():
    # 1) Build hydrogen per-kg resource demands by source
    hydrogen_perkg_by_source = build_hydrogen_perkg_by_source()

    # 2) Process-level per-kg resource demands
    h2_to_nh3_perkg = get_process_resources(filenames["hydrogen_to_ammonia"])
    h2_liq_perkg = get_process_resources(filenames["hydrogen_liquefaction"])

    # 3) Per-kg resource demands for FINAL fuels (already including upstream)
    resources_by_fuel = get_resources_by_fuel(
        hydrogen_perkg_by_source,
        h2_to_nh3_perkg,
        h2_liq_perkg
    )
    print("Per-kg resource demands by fuel / hydrogen source (including upstream H2 and processes):")
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

    # Save
    full_output.to_csv("tables/total_resource_demands_iea_2050_with_h2_components.csv",
                       index=False)

    print("\nFull output including explicit hydrogen component rows:")
    print(full_output)


if __name__ == "__main__":
    main()
