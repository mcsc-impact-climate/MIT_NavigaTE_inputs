"""
Date: 250902
Author: danikam
Purpose: Calculate costs and emissions of land transport (fuel-aware + unit conversions)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, NamedTuple

import pandas as pd
import matplotlib.pyplot as plt

from common_tools import get_top_dir, get_fuel_density, get_fuel_LHV
from load_inputs import load_molecular_info, load_global_parameters

# -------------------- Constants & Globals --------------------

DBT_DENSITY = 1.045  # kg/L, from https://www.czwinschem.com.cn/product/18440.html
CAD_PER_USD_2020 = 1.3415  # https://www.bankofcanada.ca/rates/exchange/annual-average-exchange-rates/
CAD_PER_USD_2000 = 1.4847  # https://www.ofx.com/en-ca/forex-news/historical-exchange-rates/yearly-average-rates/

PIPELINE_CSV_REL = "input_fuel_pathway_data/transport/pipeline_transport_costs_emissions.csv"

top_dir = get_top_dir()
mol = load_molecular_info()
glob = load_global_parameters()

fuel_types = {
    "Hydrogen": ["liquid_hydrogen", "compressed_hydrogen"],
    "Ammonia": ["ammonia"],
    "Hydrocarbon": ["methanol", "FTdiesel", "bio_cfp", "bio_leo"],
    "Natural gas": ["lng", "lsng"],
}

# -------------------- Small utilities --------------------

DistanceRange = Tuple[float, float]

def _parse_range(r: str) -> DistanceRange:
    """Parses 'min - max' or single 'max' to a (min, max) pair. Returns (0,0) on NA/invalid."""
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

def fuel_to_type(fuel_name: str) -> str:
    f = fuel_name.strip().lower()
    for ftype, fuels in fuel_types.items():
        if any(f == x.strip().lower() for x in fuels):
            return ftype
    raise ValueError(f"Fuel '{fuel_name}' not found in fuel_types mapping.")

@lru_cache(maxsize=None)
def _fuel_density_cached(fuel_name: str) -> float:
    return float(get_fuel_density(fuel_name))

@lru_cache(maxsize=None)
def _fuel_LHV_cached(fuel_name: str) -> float:
    return float(get_fuel_LHV(fuel_name))  # MJ/kg

# -------------------- Conversions (centralized) --------------------

@dataclass(frozen=True)
class Converters:
    """All basis & unit conversions consolidated here."""

    MW_H2:  float = float(mol["MW_H2"]["value"])
    MW_NH3: float = float(mol["MW_NH3"]["value"])
    MW_DBT: float = float(mol["MW_DBT"]["value"])
    nH_H2:  float = float(mol["nH_H2"]["value"])
    nH_DBT: float = float(mol["nH_DBT"]["value"])

    USD_2000_to_2024: float = float(glob["2000_to_2024_USD"]["value"])
    USD_2020_to_2024: float = float(glob["2020_to_2024_USD"]["value"])
    NG_density_STP:   float = float(glob["NG_density_STP"]["value"])  # kg/m^3

    def fuel_basis_multiplier(self, fuel_name: str, ftype: str) -> float:
        ftype_l = ftype.strip().lower()
        if ftype_l == "ammonia":
            # per kg H2 -> per kg NH3
            return (self.MW_H2 / self.MW_NH3) * (3.0 / 2.0)
        if ftype_l == "hydrocarbon":
            rho_fuel = _fuel_density_cached(fuel_name)  # kg/L
            return (self.MW_H2 / self.MW_DBT) * (self.nH_DBT / self.nH_H2) * (DBT_DENSITY / rho_fuel)
        # hydrogen / natural gas: identity (handled elsewhere)
        return 1.0

    def currency_to_2024usd(self, unit_str: str) -> float:
        u = (unit_str or "").lower()
        if "2000 cad" in u:
            # 2000 CAD -> 2024 USD
            return self.USD_2000_to_2024 / CAD_PER_USD_2000
        if "2020 cad" in u:
            # 2020 CAD -> 2024 USD
            return self.USD_2020_to_2024 / CAD_PER_USD_2020
        return 1.0

    def cost_to_perkg(self, unit_str: str, is_natural_gas: bool) -> float:
        u = (unit_str or "").lower()
        if is_natural_gas and "/ m^3" in u:
            # $/m^3 -> $/kg
            return 1.0 / self.NG_density_STP
        return 1.0

    def emissions_to_kg_perkg(self, unit_str: str, fuel_name: str) -> float:
        u = (unit_str or "").lower()
        if "gco2e/gj" in u:
            # g/GJ * (GJ/kg) -> g/kg -> kg/kg
            LHV_MJkg = _fuel_LHV_cached(fuel_name)
            return (LHV_MJkg / 1000.0) / 1000.0
        if "gco2e/kg" in u:
            return 1.0 / 1000.0
        if "kgco2e/kg" in u:
            return 1.0
        return 1.0

CONV = Converters()

# -------------------- Core data prep --------------------

def _load_pipeline_csv(csv_path: Optional[str]) -> pd.DataFrame:
    path = csv_path or os.path.join(top_dir, PIPELINE_CSV_REL)
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def prepare_pipeline_df(fuel: str, csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a DataFrame filtered to the fuel's type with A/B columns converted to final units:
      A_cost, B_cost in 2024 USD / kg fuel
      A_emissions, B_emissions in kg CO2e / kg fuel
    """
    fuel_norm = fuel.strip()
    ftype = fuel_to_type(fuel_norm)
    is_ng = (ftype.lower() == "natural gas")

    df = _load_pipeline_csv(csv_path)
    if "Fuel Type" in df.columns:
        df = df[df["Fuel Type"].str.strip().str.lower() == ftype.lower()].copy()
    elif "Fuel" in df.columns:
        # Fallback: explicit per-fuel rows
        df = df[df["Fuel"].str.strip().str.lower() == fuel_norm.lower()].copy()

    if df.empty:
        src = csv_path or os.path.join(top_dir, PIPELINE_CSV_REL)
        raise ValueError(f"No pipeline rows found for fuel='{fuel_norm}' (type='{ftype}') in {src}")

    fb_mult = CONV.fuel_basis_multiplier(fuel_norm, ftype)

    # Convert in place to final units
    for idx, row in df.iterrows():
        a_cost_u = str(row.get("A_cost units", ""))
        b_cost_u = str(row.get("B_cost units", ""))
        a_em_u   = str(row.get("A_emissions units", ""))
        b_em_u   = str(row.get("B_emissions units", ""))

        costA = CONV.currency_to_2024usd(a_cost_u) * CONV.cost_to_perkg(a_cost_u, is_ng) * fb_mult
        costB = CONV.currency_to_2024usd(b_cost_u) * CONV.cost_to_perkg(b_cost_u, is_ng) * fb_mult
        emA   = CONV.emissions_to_kg_perkg(a_em_u, fuel_norm) * fb_mult
        emB   = CONV.emissions_to_kg_perkg(b_em_u, fuel_norm) * fb_mult

        if pd.notna(row.get("A_cost")):
            df.at[idx, "A_cost"] = float(row["A_cost"]) * costA
        if pd.notna(row.get("B_cost")):
            df.at[idx, "B_cost"] = float(row["B_cost"]) * costB
        if pd.notna(row.get("A_emissions")):
            df.at[idx, "A_emissions"] = float(row["A_emissions"]) * emA
        if pd.notna(row.get("B_emissions")):
            df.at[idx, "B_emissions"] = float(row["B_emissions"]) * emB

    return df

# -------------------- Calculator & Plotting --------------------

class PipelineResult(NamedTuple):
    cost_per_tonne_usd2024: float
    emissions_per_kg: float
    mode: str

def calculate_land_transport_cost_emissions(
    fuel: str,
    distance_km: float,
    make_plots: bool = False,
    csv_path: Optional[str] = None,
) -> PipelineResult:
    """
    Pipeline-only calculator (cheapest row at distance).
    Returns:
      - cost_per_tonne_usd2024 (2024 USD / t fuel)
      - emissions_per_kg (kg CO2e / kg fuel)
      - mode ('pipeline')
    """
    df = prepare_pipeline_df(fuel=fuel, csv_path=csv_path)

    best_cost_per_kg = None
    best_emis_per_kg = None

    for _, row in df.iterrows():
        try:
            A_c = float(row["A_cost"]); B_c = float(row["B_cost"])
            A_e = float(row["A_emissions"]); B_e = float(row["B_emissions"])
        except Exception:
            continue

        cost_per_kg = A_c * distance_km + B_c
        emis_per_kg = A_e * distance_km + B_e

        if best_cost_per_kg is None or cost_per_kg < best_cost_per_kg:
            best_cost_per_kg = cost_per_kg
            best_emis_per_kg = emis_per_kg

    if best_cost_per_kg is None:
        raise ValueError(f"Unable to compute pipeline cost/emissions for fuel='{fuel}' at {distance_km} km.")

    if make_plots:
        plot_transport_costs_emissions(fuel=fuel, df_ready=df)

    return PipelineResult(
        cost_per_tonne_usd2024=best_cost_per_kg * 1000.0,  # per kg -> per tonne
        emissions_per_kg=best_emis_per_kg,
        mode="pipeline",
    )

def plot_transport_costs_emissions(
    fuel: str,
    df_ready: Optional[pd.DataFrame] = None,
    csv_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
    dpi: int = 160,
) -> None:
    """
    Plot lines using already-converted DataFrame (df_ready). If not provided, converts on the fly.
    Cost plot:   2024 USD / t fuel
    Emissions:   kg CO2e / kg fuel
    """
    df = df_ready if df_ready is not None else prepare_pipeline_df(fuel=fuel, csv_path=csv_path)

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
            ys = A * xs + B
            if A_col == "A_cost":
                ys = ys * 1000.0  # $/kg -> $/t

            ax.plot(xs, ys)
            x_min_global = min(x_min_global, x0)
            x_max_global = max(x_max_global, x1)

        ax.set_xlim(left=max(0.0, x_min_global), right=x_max_global)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel(y_label)
        ax.grid(True, linewidth=0.4, alpha=0.4)
        ax.set_title(title)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.tight_layout()
            fname_png = os.path.join(save_dir, outfile)
            plt.savefig(fname_png, dpi=dpi)
            plt.savefig(fname_png.replace(".png", ".pdf"))
        if show:
            plt.tight_layout()
            plt.show()
        else:
            plt.close()

    title_fuel = fuel.replace("_", " ")
    outdir = save_dir or None
    _plot_lines("A_cost", "B_cost",
                f"{title_fuel} pipeline transport cost",
                f"{fuel}_pipeline_transport_cost.png" if outdir else None,
                "2024 USD / t fuel")
    _plot_lines("A_emissions", "B_emissions",
                f"{title_fuel} pipeline transport emissions",
                f"{fuel}_pipeline_transport_emissions.png" if outdir else None,
                "kg CO₂e / kg fuel")
