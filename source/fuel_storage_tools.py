"""
Date: 2521125
Purpose: Implement HyCAT methodology from Ibrahim et al, 2025 to calculate costs and emissions to store fuel at port.
"""

from dataclasses import dataclass
from typing import Literal
import pandas as pd
from common_tools import get_fuel_density, get_top_dir
from load_inputs import load_global_parameters

top_dir = get_top_dir()

# Constants
L_PER_CBM = 1000

fuel_types = {
    "liquid_hydrogen": "Hydrogen",
    "compressed_hydrogen": "Hydrogen",
    "ammonia": "Ammonia",
    "methanol": "Hydrocarbon",
    "FTdiesel": "Hydrocarbon",
    "bio_cfp": "Hydrocarbon",
    "bio_leo": "Hydrocarbon",
    "lng": "Natural gas",
    "lsng": "Natural gas"
}

BasisType = Literal["TPC", "TOC", "UNIT_TOC_REF"]

DEFAULT_STORAGE_TIME_DAYS = 5.0
FCR = 0.0665                # Fixed charge rate from HyCAT model, %/year
CF = 0.9                    # Capacity factor from HyCAT model
TPC_TO_TOC_FACTOR = 1.2     # Factor to convert from total plant cost to total overnight cost
STORAGE_TIME_DAYS = 5       # Number of days the fuel is assumed to be stored in the tank
FIXED_OPEX_FRACTION = 0.04  # Fraction of TPC per year assigned to fixed OpEx
VAR_OPEX_FRACTION = 0       # Fraction of TPC per year assigned to variable OpEx

glob = load_global_parameters()

fuel_specific_params = {
    "Hydrogen": {
        "cost_per_tank_usd": 379.5e6, # Cost per storage tank, in 2022 USD, from HyCAT model
        "cbm_per_tank_usd": 50000,  # Capacity per storage tank, in m^3, from HyCAT model
        "SEC": 0.198,               # Specific energy consumption of fuel in tank, from HyCAT model
        "CEPCI": 1.47,               # Chemical engineering plant cost index, from HyCAT
        "location_factor_to_USA": 0.77  # Location factor needed to get the base unit TOC to USA, from HyCAT
    },
    "Ammonia": {
        "cost_per_tank_usd": 157.3e6, # Cost per storage tank, in 2022 USD, from HyCAT model
        "cbm_per_tank_usd": 75000,  # Capacity per storage tank, in m^3, from HyCAT model
        "SEC": 0.08,               # Specific energy consumption of fuel in tank, from HyCAT model
        "CEPCI": 1.47,               # Chemical engineering plant cost index, from HyCAT
        "location_factor_to_USA": 0.77  # Location factor needed to get the base unit TOC to USA, from HyCAT
    },
    "Hydrocarbon": {
        "cost_per_tank_usd": 42e6, # Cost per storage tank, in 2022 USD, from HyCAT model
        "cbm_per_tank_usd": 67208,  # Capacity per storage tank, in m^3, from HyCAT model
        "SEC": 0.000612245,               # Specific energy consumption of fuel in tank, from HyCAT model
        "CEPCI": 1.47,               # Chemical engineering plant cost index, from HyCAT
        "location_factor_to_USA": 1.00  # Location factor needed to get the base unit TOC to USA, from HyCAT
    },
    "Natural gas": {
        "cost_per_tank_usd": 737e6, # Cost per storage tank, in 2022 USD, from HyCAT model
        "cbm_per_tank_usd": 536000,  # Capacity per storage tank, in m^3, from HyCAT model
        "SEC": 0.08,               # Specific energy consumption of fuel in tank, from HyCAT model
        "CEPCI": 1.00,               # Chemical engineering plant cost index, from HyCAT
        "location_factor_to_USA": 0.77  # Location factor needed to get the base unit TOC to USA, from HyCAT
    },
}

def compute_unit_toc_from_tank(
    cost_per_tank_usd: float,
    tank_cbm: float,
    density_kg_per_m3: float,
    cepci: float,
    location_factor_to_USA: float,
    location_factor: float,
    tpc_to_toc_factor: float,
    storage_time_days: float = DEFAULT_STORAGE_TIME_DAYS,
) -> float:
    """
    Compute Unit TOC [$/ (kg/yr)] from per-tank cost and tank volume,
    assuming DEFAULT 5-day storage time for all fuels.

    U = CEPCI * LF * φ * (C_tank * τ_yr) / (V_tank * ρ)

    where τ_yr = 5/365 (default), tank volume V_tank in m^3,
    ρ is fuel density in kg/m^3.
    """
    tau_years = storage_time_days / 365.0

    # denominator is tank capacity in kg
    tank_mass_capacity_kg = tank_cbm * density_kg_per_m3

    U = (
        cepci
        * location_factor_to_USA
        * location_factor
        * tpc_to_toc_factor
        * cost_per_tank_usd
        * tau_years
        / tank_mass_capacity_kg
    )

    return U  # units = $/(kg/yr)


# ===========================================================================
# 3. Storage Technology Parameters
# ===========================================================================
@dataclass
class StorageTechParams:
    fcr: float                     # financial capital recovery factor [1/yr]
    capacity_factor: float         # CF, utilization fraction
    fixed_opex_fraction: float     # % of Unit TOC (annualized) as fixed OPEX
    variable_opex_fraction: float  # % of Unit TOC as variable OPEX
    sec_electric_kwh_per_kg: float # kWh/kg electricity


# ===========================================================================
# 4. Energy prices & CI
# ===========================================================================
@dataclass
class EnergyPriceAndCI:
    electricity_price_per_kwh: float
    electricity_ci_per_kwh: float

# ===========================================================================
# 5. Final per-kg storage cost and CI (unchanged except calling tank-based U)
# ===========================================================================
def storage_cost_and_ci_per_kg_from_tank(
    cost_per_tank_usd: float,
    tank_cbm: float,
    density_kg_per_m3: float,
    cepci: float,
    location_factor_to_USA: float,
    location_factor: float,
    tpc_to_toc_factor: float,
    tech: StorageTechParams,
    energy: EnergyPriceAndCI,
    storage_time_days: float = DEFAULT_STORAGE_TIME_DAYS,
):
    """
    Compute storage cost and CI per kg fuel using per-tank economics with
    default 5-day storage time (unless overridden).
    """

    # Step 1: Unit TOC [$/ (kg/yr)]
    U_2022 = compute_unit_toc_from_tank(
        cost_per_tank_usd=cost_per_tank_usd,
        tank_cbm=tank_cbm,
        density_kg_per_m3=density_kg_per_m3,
        cepci=cepci,
        location_factor_to_USA=location_factor_to_USA,
        location_factor=location_factor,
        tpc_to_toc_factor=tpc_to_toc_factor,
        storage_time_days=storage_time_days,
    )
    
    # Convert unit TOC from 2022 USD to 2024 USD
    
    U_2024 = U_2022 * float(glob["2022_to_2024_USD"]["value"])

    CF = tech.capacity_factor

    # Step 2: Capital + OPEX contributions [$/kg]
    capex_per_kg = U_2024 * tech.fcr / CF
    opex_fixed_per_kg = U_2024 * tech.fixed_opex_fraction / CF
    opex_variable_per_kg = U_2024 * tech.variable_opex_fraction / CF

    # Step 3: Energy contributions [$/kg]
    power_cost_per_kg = tech.sec_electric_kwh_per_kg * energy.electricity_price_per_kwh

    total_cost_per_kg = (
        capex_per_kg
        + opex_fixed_per_kg
        + opex_variable_per_kg
        + power_cost_per_kg
    )

    # Step 4: Emissions [kgCO2/kg]
    power_ci_per_kg = tech.sec_electric_kwh_per_kg * energy.electricity_ci_per_kwh

    total_ci_per_kg = power_ci_per_kg

    return {
        "unit_toc_per_kg_per_year": U_2024,
        "cost_per_kg": total_cost_per_kg,
        "breakdown_cost": {
            "capex": capex_per_kg,
            "opex_fixed": opex_fixed_per_kg,
            "opex_variable": opex_variable_per_kg,
            "power": power_cost_per_kg,
        },
        "ci_per_kg": total_ci_per_kg,
        "breakdown_ci": {
            "power": power_ci_per_kg,
        },
    }

# DME: Need to convert everything to 2024 USD
def calculate_fuel_storage_cost_emissions(
    fuel: str,
    origin_country: str,
) -> float:
    """
    Calculate storage cost in $/kg fuel for a given fuel and origin country,
    using HyCAT-inspired tank-based economics and a default 5-day storage time.
    """

    # Map fuel -> broad fuel type (Hydrogen, Ammonia, Hydrocarbon, Natural gas)
    fuel_type = fuel_types[fuel]
    fuel_params = fuel_specific_params[fuel_type]

    # Load regional TEA inputs to get installation location factor
    regional_tea_inputs = pd.read_csv(
        f"{top_dir}/input_fuel_pathway_data/regional_TEA_inputs.csv",
        index_col="Region",
    )

    # Location factor for the origin country (average of low/high factors)
    location_factor_low = regional_tea_inputs.loc[
        origin_country, "Lower Installation Cost Factor (low)"
    ]
    location_factor_high = regional_tea_inputs.loc[
        origin_country, "Upper Installation Cost Factor (high)"
    ]
    location_factor = (location_factor_low + location_factor_high) / 2.0
    
    # Grid electricity price, in 2024 USD / kWh
    electricity_price_per_kwh = regional_tea_inputs.loc[
        origin_country, "Grid Electricity price [2024$/kWh]"
    ]
    
    # Grid electricity CI, in kg CO2e / kWh
    electricity_ci_per_kwh = regional_tea_inputs.loc[
        origin_country, "Grid Electricity Emissions [kgCO2e/kWh]"
    ]

    # Fuel density: common_tools.get_fuel_density returns kg/L, convert to kg/m^3
    density_kg_per_l = get_fuel_density(fuel)
    density_kg_per_m3 = density_kg_per_l * L_PER_CBM

    # Build storage tech parameters (HyCAT-like defaults)
    storage_params = StorageTechParams(
        fcr=FCR,
        capacity_factor=CF,
        fixed_opex_fraction=FIXED_OPEX_FRACTION,
        variable_opex_fraction=VAR_OPEX_FRACTION,
        sec_electric_kwh_per_kg=fuel_params["SEC"],
    )

    # Energy prices & CI (you pass these in)
    energy = EnergyPriceAndCI(
        electricity_price_per_kwh=electricity_price_per_kwh,
        electricity_ci_per_kwh=electricity_ci_per_kwh,
    )

    # Call the core model
    result = storage_cost_and_ci_per_kg_from_tank(
        cost_per_tank_usd=fuel_params["cost_per_tank_usd"],
        tank_cbm=fuel_params["cbm_per_tank_usd"],
        density_kg_per_m3=density_kg_per_m3,
        cepci=fuel_params["CEPCI"],
        location_factor_to_USA=fuel_params["location_factor_to_USA"],
        location_factor=location_factor,
        tpc_to_toc_factor=TPC_TO_TOC_FACTOR,
        tech=storage_params,
        energy=energy,
        storage_time_days=DEFAULT_STORAGE_TIME_DAYS,
    )
    
    # Return just the storage cost in $/kg fuel
    return result["cost_per_kg"], result["ci_per_kg"]

    
## Main function to test, should be commented out in general
#def main():
#    fuel_storage_cost, fuel_storage_emissions = calculate_fuel_storage_cost_emissions("liquid_hydrogen", "United States")
#    print(fuel_storage_cost)
#
#main()

    
    
