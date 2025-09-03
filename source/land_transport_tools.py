"""
Date: 250902
Author: danikam
Purpose: Calculate costs and emissions of land transport (fuel-aware + unit conversions)
"""

import os
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common_tools import get_top_dir, get_fuel_density, get_fuel_LHV
from load_inputs import load_molecular_info, load_global_parameters

DBT_DENSITY = 1.045  # kg/L, from https://www.czwinschem.com.cn/product/18440.html
CAD_PER_USD_2020 = 1.3415 # From https://www.bankofcanada.ca/rates/exchange/annual-average-exchange-rates/
CAD_PER_USD_2000 = 1.4847 # From https://www.ofx.com/en-ca/forex-news/historical-exchange-rates/yearly-average-rates/

top_dir = get_top_dir()
mol = load_molecular_info()
glob = load_global_parameters()

fuel_types = {
    "Hydrogen": ["liquid_hydrogen", "compressed_hydrogen"],
    "Ammonia": ["ammonia"],
    "Hydrocarbon": ["methanol", "FTdiesel", "bio_cfp", "bio_leo"],
    "Natural gas": ["lng", "lsng"]
}


def _parse_range(r: str) -> Tuple[float, float]:
    if pd.isna(r):
        return (0.0, 0.0)
    s = str(r).strip()
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*$", s)
    if m:
        v = float(m.group(1))
        return (0.0, v)
    return (0.0, 0.0)


def _unit_as_transported(unit_str: str) -> str:
    # Make y-axis units mode-agnostic: replace "kg H2" with "kg fuel"
    if not isinstance(unit_str, str):
        return "per kg fuel"
    return re.sub(r"\bkg\s*H2\b", "kg fuel", unit_str, flags=re.I)
    
def _fuel_to_type(fuel_name: str) -> str:
    """Map a specific fuel (e.g., 'bio_cfp') to its Fuel Type label (e.g., 'Hydrocarbon')."""
    f = fuel_name.strip().lower()
    for ftype, fuels in fuel_types.items():
        if any(f == x.strip().lower() for x in fuels):
            return ftype
    raise ValueError(f"Fuel '{fuel_name}' not found in fuel_types mapping.")
    
# --- Molecular & density-based multipliers for pipeline cost and emissions basis ---
def _fuel_basis_mult(fuel_name: str, ftype: str) -> float:
    """
    Pipeline cost and emissions basis conversion:
      - Ammonia: per kg H2 -> per kg NH3 = (MW_H2/MW_NH3) * (3/2)
      - Hydrocarbon: per kg H2 -> per kg fuel =
            (MW_H2/MW_DBT) * (nH_DBT/nH_H2) * (DBT_DENSITY / fuel_density)
      - Hydrogen / Natural gas: no extra H2->fuel mass-basis multiplier here
        (NG is handled via $/m3->$/kg elsewhere).
    """
    ftype_l = ftype.strip().lower()
    MW_H2  = float(mol["MW_H2"]["value"])
    MW_NH3 = float(mol["MW_NH3"]["value"])
    MW_DBT = float(mol["MW_DBT"]["value"])
    nH_H2  = float(mol["nH_H2"]["value"])
    nH_DBT = float(mol["nH_DBT"]["value"])

    if ftype_l == "ammonia":
        conversion_factor = (MW_H2 / MW_NH3) * (3.0 / 2.0)
        print(f"Scaling ammonia costs and emissions by {conversion_factor} to convert from 'per kg H2' to 'per kg NH3")
        return conversion_factor

    if ftype_l == "hydrocarbon":
        rho_fuel = float(get_fuel_density(fuel_name))  # kg/L
        conversion_factor = (MW_H2 / MW_DBT) * (nH_DBT / nH_H2) * (DBT_DENSITY / rho_fuel)
        print(f"Scaling {fuel_name} costs and emissions by {conversion_factor} to convert from 'per kg H2' to 'per kg {fuel_name}")
        return conversion_factor

    print(f"No cost basis conversion for fuel {fuel_name}")
    return 1.0

def plot_transport_costs_emissions(
    fuel: str,
    csv_path: str = None,
    save_dir: Optional[str] = None,
    show: bool = True,
    dpi: int = 160,
):
    """
    Plot pipeline-only transport for a specific fuel, with final units that match the calculator:
      - Cost plot:   2024 USD / t fuel
      - Emissions:   kg CO2e / kg fuel
    The function:
      * filters the pipeline CSV by Fuel Type (via the global 'fuel_types' mapping),
      * converts A/B coefficients to final units for plotting,
      * draws lines over each row's distance coverage.
    """
    fuel_norm = fuel.strip()

    # Default to the pipeline CSV
    if csv_path is None:
        csv_path = f"{top_dir}/input_fuel_pathway_data/transport/pipeline_transport_costs_emissions.csv"

    # --- Load and filter by Fuel Type ---
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    def _fuel_to_type(fuel_name: str) -> str:
        f = fuel_name.strip().lower()
        for ftype, fuels in fuel_types.items():
            if any(f == x.strip().lower() for x in fuels):
                return ftype
        raise ValueError(f"Fuel '{fuel_name}' not found in fuel_types mapping.")

    ftype = _fuel_to_type(fuel_norm)
    if "Fuel Type" in df.columns:
        df = df[df["Fuel Type"].str.strip().str.lower() == ftype.strip().lower()].copy()
    elif "Fuel" in df.columns:
        df = df[df["Fuel"].str.strip().str.lower() == fuel_norm.strip().lower()].copy()
    if df.empty:
        raise ValueError(f"No pipeline rows found for fuel='{fuel_norm}' (type='{ftype}') in {csv_path}")

    fuel_type_is_ng = (ftype.strip().lower() == "natural gas")

    fb_mult = _fuel_basis_mult(fuel_norm, ftype)

    def _currency_to_2024usd_factor(unit_str: str) -> float:
        u = (unit_str or "").lower()
        if "2000 cad" in u:
            # 2000 CAD -> 2024 USD using 2000_to_2024_USD (USD inflation) and CAD_PER_USD_2000
            return float(glob["2000_to_2024_USD"]["value"]) / CAD_PER_USD_2000
        if "2020 cad" in u:
            # 2020 CAD -> 2024 USD using 2020_to_2024_USD and CAD_PER_USD_2020
            return float(glob["2020_to_2024_USD"]["value"]) / CAD_PER_USD_2020
        return 1.0  # assume already 2024 USD

    def _cost_to_perkg_factor(unit_str: str) -> float:
        u = (unit_str or "").lower()
        # Natural gas pipeline may be specified per m^3
        if "/ m^3" in u and fuel_type_is_ng:
            rho_stp = float(glob["NG_density_STP"]["value"])  # kg/m^3
            conversion_factor = 1.0 / rho_stp  # $/m^3 -> $/kg
            print(f"Scaling NG pipeline cost by {conversion_factor} to convert from $ / m^3 to $ / kg")
            return conversion_factor
        return 1.0  # assume already per kg

    def _emissions_to_kg_perkg_factor(unit_str: str) -> float:
        u = (unit_str or "").lower()
        if "gco2e/gj" in u:
            # g/GJ * (GJ/kg) -> g/kg; then /1000 -> kg/kg
            LHV_MJkg = float(get_fuel_LHV(fuel_norm))  # MJ/kg
            return (LHV_MJkg / 1000.0) / 1000.0
        if "gco2e/kg" in u:
            return 1.0 / 1000.0
        if "kgco2e/kg" in u:
            return 1.0
        return 1.0  # assume already kg/kg

    # --- Convert each row's coefficients to final plotting units ---
    df = df.copy()
    for idx, row in df.iterrows():
        a_cost_u = str(row.get("A_cost units", ""))
        b_cost_u = str(row.get("B_cost units", ""))
        a_em_u   = str(row.get("A_emissions units", ""))
        b_em_u   = str(row.get("B_emissions units", ""))

        # COST: currency -> 2024 USD; mass basis -> per kg (incl. NG m^3->kg); fuel basis (Ammonia/Hydrocarbon)
        cost_mult_A = _currency_to_2024usd_factor(a_cost_u) * _cost_to_perkg_factor(a_cost_u) * fb_mult
        cost_mult_B = _currency_to_2024usd_factor(b_cost_u) * _cost_to_perkg_factor(b_cost_u) * fb_mult

        # EMISSIONS: normalize to kg/kg
        emis_mult_A = _emissions_to_kg_perkg_factor(a_em_u) * fb_mult
        emis_mult_B = _emissions_to_kg_perkg_factor(b_em_u) * fb_mult

        if pd.notna(row.get("A_cost")):
            df.at[idx, "A_cost"] = float(row["A_cost"]) * cost_mult_A
        if pd.notna(row.get("B_cost")):
            df.at[idx, "B_cost"] = float(row["B_cost"]) * cost_mult_B
        if pd.notna(row.get("A_emissions")):
            df.at[idx, "A_emissions"] = float(row["A_emissions"]) * emis_mult_A
        if pd.notna(row.get("B_emissions")):
            df.at[idx, "B_emissions"] = float(row["B_emissions"]) * emis_mult_B

    # --- Plot helper ---
    def _plot_lines(A_col: str, B_col: str, title: str, outfile: Optional[str], y_label: str):
        plt.figure(figsize=(8.8, 5.6))
        ax = plt.gca()
        x_min_global, x_max_global = float("inf"), 0.0

        for _, row in df.iterrows():
            try:
                A = float(row[A_col]); B = float(row[B_col])
            except Exception:
                continue

            x0, x1 = _parse_range(row.get("Distance coverage (km)", ""))
            if x1 <= x0:
                x0, x1 = 0.0, 6000.0

            xs = pd.Series([x0, x1], dtype=float)

            # For cost plot convert per-kg -> per-tonne (×1000)
            if A_col == "A_cost":
                ys = (A * xs + B) * 1000.0
            else:
                ys = A * xs + B

            #label = f"{str(row.get('Mode','Pipeline')).strip()}"
            ax.plot(xs, ys) #, label=label)
            x_min_global = min(x_min_global, x0)
            x_max_global = max(x_max_global, x1)

        ax.set_xlim(left=max(0.0, x_min_global), right=x_max_global)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel(y_label)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        #ax.legend(loc="best", fontsize=9, frameon=False)
        ax.set_title(title)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, outfile), dpi=dpi)
            plt.savefig(os.path.join(save_dir, outfile.replace(".png", ".pdf")))
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    # COSTS (2024 USD / t fuel)
    _plot_lines(
        A_col="A_cost",
        B_col="B_cost",
        title=f"{fuel.replace('_', ' ')} pipeline transport cost",
        outfile=f"{fuel}_pipeline_transport_cost.png" if save_dir else None,
        y_label="2024 USD / t fuel",
    )

    # EMISSIONS (kg CO2e / kg fuel)
    _plot_lines(
        A_col="A_emissions",
        B_col="B_emissions",
        title=f"{fuel.replace('_', ' ')} pipeline transport emissions",
        outfile=f"{fuel}_pipeline_transport_emissions.png" if save_dir else None,
        y_label="kg CO₂e / kg fuel",
    )

    
def calculate_land_transport_cost_emissions(
    fuel: str,
    distance_km: float,
    make_plots: bool = False,
) -> Tuple[float, float, str]:
    """
    Pipeline-only calculator.
    For a given fuel and distance:
      * Filters the pipeline CSV by Fuel Type (using 'fuel_types').
      * Converts A/B coefficients to:
          - Cost:   2024 USD / kg fuel
          - Emiss.: kg CO2e / kg fuel
      * Evaluates y = A*x + B at 'distance_km' for each matching row.
      * Picks the lowest-cost row.
      * Returns (cost_2024USD_per_tonne_fuel, emissions_kgCO2e_per_kg_fuel, 'pipeline').
      * If make_plots=True, produces matching plots.
    """
    fuel_norm = fuel.strip()
    csv_path = f"{top_dir}/input_fuel_pathway_data/transport/pipeline_transport_costs_emissions.csv"

    # --- Load and filter by Fuel Type ---
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    def _fuel_to_type(fuel_name: str) -> str:
        f = fuel_name.strip().lower()
        for ftype, fuels in fuel_types.items():
            if any(f == x.strip().lower() for x in fuels):
                return ftype
        raise ValueError(f"Fuel '{fuel_name}' not found in fuel_types mapping.")

    ftype = _fuel_to_type(fuel_norm)
    if "Fuel Type" in df.columns:
        df = df[df["Fuel Type"].str.strip().str.lower() == ftype.strip().lower()].copy()
    elif "Fuel" in df.columns:
        df = df[df["Fuel"].str.strip().str.lower() == fuel_norm.strip().lower()].copy()
    if df.empty:
        raise ValueError(f"No pipeline rows found for fuel='{fuel_norm}' (type='{ftype}') in {csv_path}")

    fuel_type_is_ng = (ftype.strip().lower() == "natural gas")

    fb_mult = _fuel_basis_mult(fuel_norm, ftype)

    def _currency_to_2024usd_factor(unit_str: str) -> float:
        u = (unit_str or "").lower()
        if "2000 cad" in u:
            return float(glob["2000_to_2024_USD"]["value"]) / CAD_PER_USD_2000
        if "2020 cad" in u:
            return float(glob["2020_to_2024_USD"]["value"]) / CAD_PER_USD_2020
        return 1.0

    def _cost_to_perkg_factor(unit_str: str) -> float:
        u = (unit_str or "").lower()
        if "/ m^3" in u and fuel_type_is_ng:
            rho_stp = float(glob["NG_density_STP"]["value"])  # kg/m^3
            conversion_factor = 1.0 / rho_stp  # $/m^3 -> $/kg
            print(f"Scaling NG pipeline cost by {conversion_factor} to convert from $ / m^3 to $ / kg")
            return conversion_factor
        return 1.0

    def _emissions_to_kg_perkg_factor(unit_str: str) -> float:
        u = (unit_str or "").lower()
        if "gco2e/gj" in u:
            LHV_MJkg = float(get_fuel_LHV(fuel_norm))  # MJ/kg
            conversion_factor = (LHV_MJkg / 1000.0) / 1000.0 # g/GJ -> kg/kg
            print(f"Scaling emissions of fuel {fuel} by {conversion_factor} to convert from g/GJ to kg/kg")
            return conversion_factor
        if "gco2e/kg" in u:
            return 1.0 / 1000.0
        if "kgco2e/kg" in u:
            return 1.0
        return 1.0

    # --- Convert coefficients to final calculator units ---
    df = df.copy()
    for idx, row in df.iterrows():
        a_cost_u = str(row.get("A_cost units", ""))
        b_cost_u = str(row.get("B_cost units", ""))
        a_em_u   = str(row.get("A_emissions units", ""))
        b_em_u   = str(row.get("B_emissions units", ""))

        cost_mult_A = _currency_to_2024usd_factor(a_cost_u) * _cost_to_perkg_factor(a_cost_u) * fb_mult
        cost_mult_B = _currency_to_2024usd_factor(b_cost_u) * _cost_to_perkg_factor(b_cost_u) * fb_mult
        emis_mult_A = _emissions_to_kg_perkg_factor(a_em_u) * fb_mult
        emis_mult_B = _emissions_to_kg_perkg_factor(b_em_u) * fb_mult

        if pd.notna(row.get("A_cost")):
            df.at[idx, "A_cost"] = float(row["A_cost"]) * cost_mult_A
        if pd.notna(row.get("B_cost")):
            df.at[idx, "B_cost"] = float(row["B_cost"]) * cost_mult_B

        if pd.notna(row.get("A_emissions")):
            df.at[idx, "A_emissions"] = float(row["A_emissions"]) * emis_mult_A
        if pd.notna(row.get("B_emissions")):
            df.at[idx, "B_emissions"] = float(row["B_emissions"]) * emis_mult_B

    # --- Evaluate at distance and choose the cheapest pipeline row ---
    best = None  # (cost_per_kg_2024USD, emissions_kg_per_kg, idx)
    for idx, row in df.iterrows():
        try:
            A_c = float(row["A_cost"]); B_c = float(row["B_cost"])
            A_e = float(row["A_emissions"]); B_e = float(row["B_emissions"])
        except Exception:
            continue

        cost_per_kg_usd2024 = A_c * distance_km + B_c
        emis_kg_per_kg = A_e * distance_km + B_e

        if (best is None) or (cost_per_kg_usd2024 < best[0]):
            best = (cost_per_kg_usd2024, emis_kg_per_kg, idx)

    if best is None:
        raise ValueError(f"Unable to compute pipeline cost/emissions for fuel='{fuel_norm}' at {distance_km} km.")

    cost_per_tonne_usd2024 = best[0] * 1000.0  # per kg -> per tonne
    emissions_per_kg = best[1]
    mode_out = "pipeline"

    # Optional plots
    if make_plots:
        plot_transport_costs_emissions(
            fuel=fuel_norm,
            csv_path=csv_path,
            save_dir=f"{top_dir}/plots/transport",
            show=False,
            dpi=160,
        )

    return cost_per_tonne_usd2024, emissions_per_kg



if __name__ == "__main__":
    fuels = ["liquid_hydrogen", "compressed_hydrogen", "ammonia", "methanol", "FTdiesel", "bio_cfp", "bio_leo", "lng", "lsng"]
    for fuel in fuels:
        cost_t_usd2024, em_kgkg = calculate_land_transport_cost_emissions(fuel, distance_km=1000, make_plots=True)
        print(f"{fuel=} {cost_t_usd2024=:.2f} USD/t, {em_kgkg=:.3f} kgCO2e/kg")

