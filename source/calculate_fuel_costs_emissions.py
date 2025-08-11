"""
Date: Aug 21, 2024. Updated on June 11 2025 to improve modularity and align with best coding practices.
Purpose: Prepare .csv files contained in input_fuel_pathway_data using consistent assumptions.
"""

import os
from typing import Dict
from dataclasses import dataclass
from typing import Callable
import pandas as pd
import argparse
import json

# ---------------------------------------------------------------------------
# One-time loaders wrapped in a helper – *nothing else calls these yet*
# ---------------------------------------------------------------------------
from common_tools import ensure_directory_exists, get_top_dir
from load_inputs import (
    load_global_parameters,
    load_molecular_info,
    load_technology_info,
)


def build_context() -> Dict[str, Dict]:
    """
    Loads top-level inputs once and pre-computes all pathway-specific
    quantities (with full explanatory comments).
    """
    ctx = {
        "top_dir": get_top_dir(),
        "glob": load_global_parameters(),
        "molecular_info": load_molecular_info(),
        "tech": load_technology_info(),
        "derived": {},
    }

    # Short aliases ----------------------------------------------------
    G = ctx["glob"]
    M = ctx["molecular_info"]
    T = ctx["tech"]
    print(T)
    D = ctx["derived"]
    top_dir = ctx["top_dir"]

    # ------------------------------------------------------------------
    # Natural-gas recovery and liquefaction from GREET 2024
    # ------------------------------------------------------------------
    ng_info = pd.read_csv(
        os.path.join(
            ctx["top_dir"], "input_fuel_pathway_data", "lng_inputs_GREET_processed.csv"
        ),
        index_col="Stage",
    )

    D["NG_NG_demand_kg"] = ng_info.loc["Production", "NG Consumption (kg/kg)"]
    D["NG_NG_demand_GJ"] = ng_info.loc["Production", "NG Consumption (GJ/kg)"]
    D["NG_water_demand"] = ng_info.loc["Production", "Water Consumption (m^3/kg)"]
    D["NG_elect_demand"] = ng_info.loc["Production", "Electricity Consumption (kWh/kg)"]
    D["NG_CO2_emissions"] = ng_info.loc["Production", "CO2 Emissions (kg/kg)"]
    D["NG_CH4_leakage"] = ng_info.loc["Production", "CH4 Emissions (kg/kg)"]

    D["NG_liq_NG_demand_kg"] = ng_info.loc["Liquefaction", "NG Consumption (kg/kg)"]
    D["NG_liq_NG_demand_GJ"] = ng_info.loc["Liquefaction", "NG Consumption (GJ/kg)"]
    D["NG_liq_water_demand"] = ng_info.loc["Liquefaction", "Water Consumption (m^3/kg)"]
    D["NG_liq_elect_demand"] = ng_info.loc[
        "Liquefaction", "Electricity Consumption (kWh/kg)"
    ]
    D["NG_liq_CO2_emissions"] = ng_info.loc["Liquefaction", "CO2 Emissions (kg/kg)"]
    D["NG_liq_CH4_leakage"] = ng_info.loc["Liquefaction", "CH4 Emissions (kg/kg)"]

    # NH3 from H2 stoichiometry
    D["NH3_H2_demand"] = 3 / 2 * M["MW_H2"]["value"] / M["MW_NH3"]["value"]

    D["NH3_elect_demand"] = (
        T["NH3"]["elect_demand"]["value"]
        - T["H2_LTE"]["elect_demand"]["value"] * D["NH3_H2_demand"]
    )
    D["NH3_water_demand"] = (
        T["NH3"]["water_demand_LTE"]["value"]
        - T["H2_LTE"]["water_demand"]["value"] * D["NH3_H2_demand"]
    )
    D["NH3_base_CapEx"] = (
        T["NH3"]["base_CapEx_LTE"]["value"]
        - T["H2_LTE"]["base_CapEx"]["value"] * D["NH3_H2_demand"]
    )
    D["NH3_employees"] = (
        T["NH3"]["employees_LTE"]["value"] - T["H2_LTE"]["employees"]["value"]
    )

    # BEC emissions
    def calculate_BEC_upstream_emission_rate(
        filename=f"{top_dir}/input_fuel_pathway_data/BEC_upstream_emissions_GREET.csv",
    ):
        df = pd.read_csv(filename)
        df["Upstream emissions (kg CO2e / kg CO2"] = (
            df["Feedstock emissions (g CO2e/mmBtu)"]
            + df["Fuel emissions (g CO2e/mmBtu)"]
        ) / df["CO2 from CCS"]
        return df["Upstream emissions (kg CO2e / kg CO2"].mean()

    D["BEC_CO2_price"] = G["BEC_CO2_price"]["value"]
    D["BEC_CO2_upstream_emissions"] = calculate_BEC_upstream_emission_rate()

    # DAC emissions and resources
    def calculate_DAC_upstream_resources_emissions():
        upstream = pd.read_csv(
            f"{top_dir}/input_fuel_pathway_data/DAC_upstream_electricity_NG.csv"
        )
        upstream_elec = (
            upstream["Electricity for CO2 capture (MJ/MT-CO2)"][0]
            + upstream["Electricity for CO2 compression at the CO2 source (MJ/MT-CO2)"][
                0
            ]
        )
        upstream_elec /= 3.6 * 1000

        upstream_NG = upstream["Natural gas for CO2 capture (MJ/MT-CO2)"][0]
        upstream_NG /= 1000 * 1000

        materials = pd.read_csv(
            f"{top_dir}/input_fuel_pathway_data/DAC_material_reqs.csv"
        )
        GAL_PER_CBM = 264.172
        BTU_PER_MJ = 947.817
        BTU_PER_MMBTU = 1e6
        KG_PER_TON = 907.185

        materials["Water"] = (
            materials["Water consumption (gals/ton)"] / (GAL_PER_CBM * KG_PER_TON)
        ) * (materials["kg / ton CO2"] / KG_PER_TON)
        materials["NG"] = (
            materials["NG consumption (mmBtu/ton)"]
            * BTU_PER_MMBTU
            / (BTU_PER_MJ * 1000 * KG_PER_TON)
        ) * (materials["kg / ton CO2"] / KG_PER_TON)
        materials["GHG"] = (
            materials["GHG emissions (g CO2e/ton)"] / (1000 * KG_PER_TON)
        ) * (materials["kg / ton CO2"] / KG_PER_TON)

        return {
            "emissions": materials["GHG"].sum(),
            "elect": upstream_elec,
            "NG": upstream_NG + materials["NG"].sum(),
            "water": materials["Water"].sum(),
        }

    dac = calculate_DAC_upstream_resources_emissions()
    D["DAC_CO2_price"] = G["DAC_CO2_price"]["value"]
    D["DAC_CO2_upstream_emissions"] = dac["emissions"]
    D["DAC_CO2_upstream_NG"] = dac["NG"]
    D["DAC_CO2_upstream_water"] = dac["water"]
    D["DAC_CO2_upstream_elect"] = dac["elect"]

    # ------------------------------------------------------------------
    # DERIVED VALUES for hydrogen pathways
    # ------------------------------------------------------------------

    # ── STP H₂ via SMR (no capture) ────────────────────────────────────
    # Inputs from Zang et al 2024 (SMR case)
    smr_prod = T["H2_SMR"]["hourly_prod"]["value"]  # [kg H₂/h]
    D["H2_SMR_elect_demand"] = (
        T["H2_SMR"]["elec_cons"]["value"] / smr_prod
    )  # [kWh elect/kg H₂]
    SMR_NG = (
        T["H2_SMR"]["NG_cons"]["value"]
        - T["H2_SMR"]["steam_byproduct"]["value"] / T["H2_SMR"]["boiler_eff"]["value"]
    )  # [GJ NG/hr]: NG consumption including steam displacement at 80% boiler efficiency
    D["H2_SMR_NG_demand"] = (  # [GJ NG/kg H₂]
        SMR_NG
        / smr_prod
        * (
            1 + D["NG_NG_demand_kg"]
        )  # Also account for the additional NG consumed to process and recover the NG, from GREET 2024
    )
    D["H2_SMR_water_demand"] = (
        T["H2_SMR"]["water_cons"]["value"] / smr_prod
    )  # [m³ H₂O/kg H₂]
    # Base CapEx, converted from 2019 USD to 2024 USD then amortised (From Zang et al 2024 and H2A)
    D["H2_SMR_base_CapEx"] = (
        G["2019_to_2024_USD"]["value"]
        * T["H2_SMR"]["TPC_2019"]["value"]
        / 365
        * T["H2_SMR"]["CRF"]["value"]
    )
    D["H2_SMR_onsite_emissions"] = (
        T["H2_SMR"]["emissions"]["value"] / smr_prod
    )  # [kg CO2e/kg H₂]
    D["H2_SMR_yearly_output"] = smr_prod * 24 * 365  # [kg H₂/year]

    # ── STP H₂ via ATR-CC-S (99 % capture; ATR-CC-R-OC case) ───────────
    atr_prod = T["H2_ATRCCS"]["hourly_prod"]["value"]
    D["H2_ATRCCS_elect_demand"] = (
        T["H2_ATRCCS"]["elec_cons"]["value"] / atr_prod
    )  # [kWh elect/kg H₂]
    D["H2_ATRCCS_NG_demand"] = (  # [GJ NG/kg H₂]
        T["H2_ATRCCS"]["NG_cons"]["value"]
        / atr_prod
        * (1 + D["NG_NG_demand_kg"])  # GREET 2024 uplift
    )
    D["H2_ATRCCS_water_demand"] = (
        T["H2_ATRCCS"]["water_cons"]["value"] / atr_prod
    )  # [m³ H₂O/kg H₂]
    # Amortised TPC – From Zang et al 2024 and H2A
    D["H2_ATRCCS_base_CapEx"] = (
        G["2019_to_2024_USD"]["value"]
        * T["H2_ATRCCS"]["TPC_2019"]["value"]
        / 365
        * T["H2_ATRCCS"]["CRF"]["value"]
    )
    D["H2_ATRCCS_onsite_emissions"] = (
        T["H2_ATRCCS"]["emissions"]["value"] / atr_prod
    )  # [kg CO2e/kg H₂]
    D["H2_ATRCCS_yearly_output"] = atr_prod * 24 * 365  # [kg H₂/year]

    # ── STP H₂ via SMR-CCS (96 % capture) ─────────────────────────────
    smrccs_prod = T["H2_SMRCCS"]["hourly_prod"]["value"]
    D["H2_SMRCCS_elect_demand"] = T["H2_SMRCCS"]["elec_cons"]["value"] / smrccs_prod
    D["H2_SMRCCS_NG_demand"] = (
        T["H2_SMRCCS"]["NG_cons"]["value"] / smrccs_prod * (1 + D["NG_NG_demand_kg"])
    )
    D["H2_SMRCCS_water_demand"] = T["H2_SMRCCS"]["water_cons"]["value"] / smrccs_prod
    D["H2_SMRCCS_base_CapEx"] = (
        G["2019_to_2024_USD"]["value"]
        * T["H2_SMRCCS"]["TPC_2019"]["value"]
        / 365
        * T["H2_SMRCCS"]["CRF"]["value"]
    )
    D["H2_SMRCCS_onsite_emissions"] = T["H2_SMRCCS"]["emissions"]["value"] / smrccs_prod
    D["H2_SMRCCS_yearly_output"] = smrccs_prod * 24 * 365

    # ── Low-Temperature Electrolysis (LTE) ─────────────────────────────
    D["H2_LTE_elect_demand"] = T["H2_LTE"]["elect_demand"][
        "value"
    ]  # [kWh/kg H₂] from H2A
    D["H2_LTE_NG_demand"] = T["H2_LTE"]["NG_demand"]["value"]  # [GJ/kg H₂] aux boiler
    D["H2_LTE_water_demand"] = T["H2_LTE"]["water_demand"]["value"]  # [m³/kg H₂]
    D["H2_LTE_base_CapEx"] = T["H2_LTE"]["base_CapEx"]["value"]  # [2024 $ / kg H₂ y⁻¹]
    D["H2_LTE_onsite_emissions"] = T["H2_LTE"]["onsite_emissions"][
        "value"
    ]  # [kg CO2e/kg H₂]
    D["H2_LTE_yearly_output"] = T["H2_LTE"]["yearly_output"]["value"]  # [kg H₂/year]

    # ── Biomass Gasification (BG) – lignocellulosic ────────────────────
    D["H2_BG_elect_demand"] = T["H2_BG"]["elect_demand"]["value"]
    D["H2_BG_NG_demand"] = (
        T["H2_BG"]["NG_demand"]["value"]
        * (1 + D["NG_NG_demand_kg"])  # GREET 2024 uplift
    )
    D["H2_BG_water_demand"] = T["H2_BG"]["water_demand"]["value"]
    D["H2_BG_base_CapEx"] = (
        G["2015_to_2024_USD"]["value"] * T["H2_BG"]["base_CapEx_2015"]["value"]
    )  # Base CapEx, converted from 2015 USD to 2024 USD
    D["H2_BG_onsite_emissions"] = (
        T["H2_BG"]["onsite_emissions"]["value"]
        - T["H2_BG"]["LCB_gasification_emissions"]["value"]
        * T["H2_BG"]["LCB_demand"]["value"]
    )  # Includes biogenic credit
    D["H2_BG_yearly_output"] = T["H2_BG"]["yearly_output"]["value"]

    return ctx

    # ------------------------------------------------------------------
    # DERIVED VALUES for hydrogen pathways
    # ------------------------------------------------------------------


CTX = build_context()
top_dir = CTX["top_dir"]

# ---------------------------------------------------------------------------
# static parameter table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pathway:
    """
    Read-only record of static parameters for ONE production pathway.

    Each field may be:
      * a literal (float / int), or
      * a small `lambda ctx:` that grabs numbers from the global CTX.
    """

    elect_demand: Callable[[dict], float] | float
    lcb_demand: Callable[[dict], float] | float
    ng_demand: Callable[[dict], float] | float
    water_demand: Callable[[dict], float] | float
    base_capex: Callable[[dict], float] | float
    employees: Callable[[dict], int] | int
    yearly_output: Callable[[dict], float] | float
    onsite_emiss: Callable[[dict], float] | float


PATHWAYS: Dict[str, Pathway] = {
    # ------------------------------------------------------------------
    # Steam‑methane reforming (no capture)
    # ------------------------------------------------------------------
    "SMR": Pathway(
        elect_demand=lambda c: c["derived"]["H2_SMR_elect_demand"],
        lcb_demand=lambda c: c["tech"]["H2_SMR"]["LCB_demand"]["value"],
        ng_demand=lambda c: c["derived"]["H2_SMR_NG_demand"],
        water_demand=lambda c: c["derived"]["H2_SMR_water_demand"],
        base_capex=lambda c: c["derived"]["H2_SMR_base_CapEx"],
        employees=lambda c: c["tech"]["H2_SMR"]["employees"]["value"],
        yearly_output=lambda c: c["derived"]["H2_SMR_yearly_output"],
        onsite_emiss=lambda c: c["derived"]["H2_SMR_onsite_emissions"],
    ),
    # ------------------------------------------------------------------
    # Autothermal reforming + CCS (99 % capture)  – Zang et al 2024 ATR‑CC‑R‑OC
    # ------------------------------------------------------------------
    "ATRCCS": Pathway(
        elect_demand=lambda c: c["derived"]["H2_ATRCCS_elect_demand"],
        lcb_demand=lambda c: 0.0,  # ATR‑CCS uses no biomass feed
        ng_demand=lambda c: c["derived"]["H2_ATRCCS_NG_demand"],
        water_demand=lambda c: c["derived"]["H2_ATRCCS_water_demand"],
        base_capex=lambda c: c["derived"]["H2_ATRCCS_base_CapEx"],
        employees=lambda c: c["tech"]["H2_ATRCCS"]["employees"]["value"],
        yearly_output=lambda c: c["derived"]["H2_ATRCCS_yearly_output"],
        onsite_emiss=lambda c: c["derived"]["H2_ATRCCS_onsite_emissions"],
    ),
    # ------------------------------------------------------------------
    # Steam‑methane reforming + CCS (96 % capture) – Zang et al 2024 SMR‑CCS
    # ------------------------------------------------------------------
    "SMRCCS": Pathway(
        elect_demand=lambda c: c["derived"]["H2_SMRCCS_elect_demand"],
        lcb_demand=lambda c: 0.0,
        ng_demand=lambda c: c["derived"]["H2_SMRCCS_NG_demand"],
        water_demand=lambda c: c["derived"]["H2_SMRCCS_water_demand"],
        base_capex=lambda c: c["derived"]["H2_SMRCCS_base_CapEx"],
        employees=lambda c: c["tech"]["H2_SMRCCS"]["employees"]["value"],
        yearly_output=lambda c: c["derived"]["H2_SMRCCS_yearly_output"],
        onsite_emiss=lambda c: c["derived"]["H2_SMRCCS_onsite_emissions"],
    ),
    # ------------------------------------------------------------------
    # Low‑temperature electrolysis (alkaline / PEM) – H2A baseline
    # ------------------------------------------------------------------
    "LTE": Pathway(
        elect_demand=lambda c: c["derived"]["H2_LTE_elect_demand"],
        lcb_demand=lambda c: c["tech"]["H2_LTE"]["LCB_demand"]["value"],
        ng_demand=lambda c: c["derived"]["H2_LTE_NG_demand"],
        water_demand=lambda c: c["derived"]["H2_LTE_water_demand"],
        base_capex=lambda c: c["derived"]["H2_LTE_base_CapEx"],
        employees=lambda c: c["tech"]["H2_LTE"]["employees"]["value"],
        yearly_output=lambda c: c["derived"]["H2_LTE_yearly_output"],
        onsite_emiss=lambda c: c["derived"]["H2_LTE_onsite_emissions"],
    ),
    # ------------------------------------------------------------------
    # Biomass gasification (lignocellulosic) – H2A gasifier case
    # ------------------------------------------------------------------
    "BG": Pathway(
        elect_demand=lambda c: c["derived"]["H2_BG_elect_demand"],
        lcb_demand=lambda c: c["tech"]["H2_BG"]["LCB_demand"]["value"],
        ng_demand=lambda c: c["derived"]["H2_BG_NG_demand"],
        water_demand=lambda c: c["derived"]["H2_BG_water_demand"],
        base_capex=lambda c: c["derived"]["H2_BG_base_CapEx"],
        employees=lambda c: c["tech"]["H2_BG"]["employees"]["value"],
        yearly_output=lambda c: c["derived"]["H2_BG_yearly_output"],
        onsite_emiss=lambda c: c["derived"]["H2_BG_onsite_emissions"],
    ),
    # ------------------------------------------------------------------
    # Natural gas at pipeline conditions (STP)
    # ------------------------------------------------------------------
    "NG": Pathway(
        elect_demand=lambda c: c["derived"]["NG_elect_demand"],
        lcb_demand=lambda c: 0.0,
        ng_demand=lambda c: c["glob"]["NG_HHV"]["value"]
        + c["derived"][
            "NG_NG_demand_GJ"
        ],  # Additionally account for the NG consumed in producing the NG
        water_demand=lambda c: c["derived"]["NG_water_demand"],
        base_capex=lambda c: 0.0,  # CapEx = 0 for commodity NG
        employees=lambda c: 0,
        yearly_output=lambda c: 1.0,  # dummy (unused when employees=0)
        onsite_emiss=lambda c: 0.0,  # accounted upstream
    ),
    # ------------------------------------------------------------------
    # Liquefied NG at 1 bar, 111 K – incremental over “NG”
    # ------------------------------------------------------------------
    "LNG": Pathway(
        elect_demand=lambda c: c["derived"]["NG_liq_elect_demand"],
        lcb_demand=lambda c: 0.0,
        ng_demand=lambda c: c["derived"]["NG_liq_NG_demand_GJ"],
        water_demand=lambda c: c["derived"]["NG_liq_water_demand"],
        base_capex=lambda c: c["glob"]["NG_liq"]["base_CapEx"]["value"],
        employees=lambda c: 0,  # assume incremental block has no labour
        yearly_output=lambda c: 1.0,
        onsite_emiss=lambda c: (
            c["derived"]["NG_liq_CO2_emissions"]
            + c["derived"]["NG_liq_CH4_leakage"] * c["glob"]["NG_GWP"]["value"]
        ),
    ),
    # ------------------------------------------------------------------
    # Synthetic NG (pipeline conditions) – H2 + CO2 → CH4 core block
    # ------------------------------------------------------------------
    "SNG": Pathway(
        elect_demand=lambda c: c["tech"]["SNG"]["elect_demand"]["value"],
        lcb_demand=lambda c: 0.0,
        ng_demand=lambda c: 0.0,  # no purchased NG in synthesis
        water_demand=lambda c: c["tech"]["SNG"]["water_demand"]["value"],
        base_capex=lambda c: c["tech"]["SNG"]["base_CapEx_2016"]["value"] * c["glob"]["2016_to_2024_USD"]["value"],
        employees=lambda c: c["tech"]["SNG"]["employees"]["value"],
        yearly_output=lambda c: c["tech"]["SNG"]["yearly_output"]["value"],
        onsite_emiss=lambda c: c["tech"]["SNG"]["onsite_emissions"]["value"],
    ),
    # ------------------------------------------------------------------
    # Liquefied SNG (LSNG) – incremental liquefaction over SNG
    # ------------------------------------------------------------------
    "LSNG": Pathway(
        elect_demand=lambda c: c["derived"]["NG_liq_elect_demand"],
        lcb_demand=lambda c: 0.0,
        ng_demand=lambda c: c["derived"]["NG_liq_NG_demand_GJ"],
        water_demand=lambda c: c["derived"]["NG_liq_water_demand"],
        # We'll compute liquefaction CapEx explicitly in the LSNG function
        base_capex=lambda c: 0.0,
        employees=lambda c: 0,
        yearly_output=lambda c: 1.0,
        onsite_emiss=lambda c: (
            c["derived"]["NG_liq_CO2_emissions"]
            + c["derived"]["NG_liq_CH4_leakage"] * c["glob"]["NG_GWP"]["value"]
        ),
    ),
}


# ---------------------------------------------------------------------------
def _p(path: str, field: str):
    """
    Helper for later steps – evaluate *field* for *pathway*.

    It understands that the stored value might be:
      * a literal    → return it
      * a lambda ctx → call it with the global CTX
    """
    val = PATHWAYS[path].__dict__[field]
    return val(CTX) if callable(val) else val


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def generic_production(
    path: str,
    product: str,
    install_factor: float,
    prices: dict,
    elec_intensity: float,
    lcb_upstream_emiss: float = 0.0,
) -> tuple[float, float, float]:
    """
    Pathway-agnostic production cost & emissions calculator.
    Returns (CapEx, OpEx, emissions) per *kg fuel*.

    Only uses the four basic demands stored in PATHWAYS:
    electricity, NG, water, LCB.
    """
    elect = _p(path, "elect_demand")
    ng = _p(path, "ng_demand")
    water = _p(path, "water_demand")
    lcb = _p(path, "lcb_demand")

    # -------- CapEx ----------------------------------------------------
    capex_components = {
        f"{product} Production Facility": _p(path, "base_capex") * install_factor
    }

    # -------- OpEx -----------------------------------------------------
    opex_components = {
        f"Labor for {product} Production": CTX["glob"]["workhours_per_year"]["value"]
                    * prices["labor"]
                    * _p(path, "employees")
                    / _p(path, "yearly_output")
                    * (1 + CTX["glob"]["gen_admin_rate"]["value"]),
        f"O&M for {product} Production": (CTX["glob"]["op_maint_rate"]["value"]
                    + CTX["glob"]["tax_rate"]["value"])
                    * sum(capex_components.values()),
        f"Electricity for {product} Production": elect * prices["elec"],
        f"NG for {product} Production": ng * prices["ng"],
        f"Water for {product} Production": water * prices["water"],
        f"LCB for {product} Production": lcb * prices["lcb"]
    }

    # -------- Emissions -------------------------------------------------
    emiss_components = {
        f"Electricity for {product} Production": elect * elec_intensity,
        f"NG for {product} Production": ng / CTX["glob"]["NG_HHV"]["value"]    # fugitive CH4 and upstream CO2-e
        * CTX["glob"]["NG_GWP"]["value"]
        * CTX["derived"]["NG_CH4_leakage"]
        + ng / CTX["glob"]["NG_HHV"]["value"] * CTX["derived"]["NG_CO2_emissions"],
        f"Onsite Emissions for {product} Production": _p(path, "onsite_emiss"),
        f"LCB Upstream Emissions for {product} Production": lcb * lcb_upstream_emiss
    }

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components


# ---------------------------------------------------------------------------
# Generic resource-demands extractor  (electricity, LCB, NG, water, CO₂)
# ---------------------------------------------------------------------------
def generic_demands(
    path: str,
    *,
    include_elect: bool = True,
    lcb_factor: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """
    Return (elect, lcb, ng, water, co₂) *per kg fuel* for any table pathway.

    Parameters
    ----------
    path            Canonical key in `PATHWAYS`
    include_elect   Skip electricity for downstream blocks that reuse it
    lcb_factor      Useful if you later need to scale LCB for moisture, etc.
    """
    elect = _p(path, "elect_demand") if include_elect else 0.0
    lcb = _p(path, "lcb_demand") * lcb_factor
    ng = _p(path, "ng_demand")
    water = _p(path, "water_demand")
    # all STP fuels in this repo consume no external CO₂
    return elect, lcb, ng, water, 0.0


# ─────────────────────────────────────────────────────────────
# Incremental H₂ liquefaction  (adds only extra electricity)
# ─────────────────────────────────────────────────────────────
def demands_liquid_h2(H_pathway: str):
    elect, lcb, ng, water, co2 = generic_demands(H_pathway)
    elect += CTX["tech"]["H2_liq"]["elect_demand"]["value"]
    return elect, lcb, ng, water, co2


# ─────────────────────────────────────────────────────────────
# Incremental H₂ compression  (adds only extra electricity)
# ─────────────────────────────────────────────────────────────
def demands_compressed_h2(H_pathway: str):
    elect, lcb, ng, water, co2 = generic_demands(H_pathway)
    elect += CTX["tech"]["H2_comp"]["elect_demand"]["value"]
    return elect, lcb, ng, water, co2

# ----------------------------------------------------------------------
# Helper: cost / emissions for 1 kg of captured-CO₂ feedstock
# ----------------------------------------------------------------------
def _feed_co2(
    C_pathway: str,
    credit_per_kg_fuel: float,  # e.g. MW_CO2 / MW_MeOH   or  nC*MW_CO2 / MW_FTdiesel
    prices: dict,
    elect_emissions_intensity: float,
    CO2_demand: float,
):
    """Return (capex, opex, emissions) *per kg CO₂* for any fuel.

    `credit_per_kg_fuel` is the (negative) credit to apply for 1 kg of fuel
    when carbon stays sequestered in the product rather than emitted.
    """
    if C_pathway == "BEC":
        capex_components = {}
        opex_components = {
            "CO2 from BEC": CTX["glob"]["BEC_CO2_price"]["value"]
        }
        emiss_components = {
        "BEC Facility Operational Emissions": CTX["derived"]["BEC_CO2_upstream_emissions"] * CO2_demand,
        "Captured Carbon Credit": -1.0 * credit_per_kg_fuel
        }
    elif C_pathway == "DAC":
        capex_components = {}
        opex_components = {
            "CO2 from DAC": CTX["glob"]["DAC_CO2_price"]["value"]
        }
        emiss_components = {
            "Embedded Emissions of DAC Facility": CTX["derived"]["DAC_CO2_upstream_emissions"] * CO2_demand,
            "Upstream Emissions of NG Used for DAC": CTX["derived"]["DAC_CO2_upstream_NG"] * CO2_demand
                                                                    / CTX["glob"]["NG_HHV"]["value"]
                                                                    * CTX["glob"]["NG_GWP"]["value"]
                                                                    * CTX["derived"]["NG_CH4_leakage"]      # Upstream CH4 leakage
                                                                    + CTX["derived"]["DAC_CO2_upstream_NG"] * CO2_demand
                                                                    / CTX["glob"]["NG_HHV"]["value"]
                                                                    * CTX["derived"]["NG_CO2_emissions"],   # Upstream CO2 emissions to produce NG
            "Upstream Emissions of Electricity Used for DAC Production": CTX["derived"]["DAC_CO2_upstream_elect"] * elect_emissions_intensity * CO2_demand,
            "Captured Carbon Credit": -1.0 * credit_per_kg_fuel
        }

    elif C_pathway in ("SMRCCS", "ATRCCS"):
        capex_components = {}
        opex_components = {}
        emiss_components = {
            "Emissions to Operate CCS": CO2_demand,
            "Captured Carbon Credit": -1.0 * credit_per_kg_fuel
        }
    elif C_pathway == "SMR":
        capex_components = {}
        opex_components = {}
        emiss_components = {
            "Captured Carbon Credit": -1.0 * credit_per_kg_fuel
        }
    elif C_pathway == "BG":
        capex_components = {}
        opex_components = {}
        emiss_components = {}  # biogenic credit already counted
    else:
        raise ValueError(f"Unknown CO₂ pathway: {C_pathway}")

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components


def calculate_production_costs_emissions_STP_hydrogen(
    H_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    hourly_labor_rate: float,
):
    prices = {
        "water": water_price,
        "ng": NG_price,
        "lcb": LCB_price,
        "elec": elect_price,
        "labor": hourly_labor_rate,
    }
    capex, opex, emiss, capex_components, opex_components, emiss_components = generic_production(
        H_pathway,
        "H2",
        instal_factor,
        prices,
        elect_emissions_intensity,
        lcb_upstream_emiss=LCB_upstream_emissions
        
    )
    return capex, opex, emiss, capex_components, opex_components, emiss_components


# ────────────────────────────────────────────────────────────────────
# Liquid cryogenic H₂ at 1 bar
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_liquid_hydrogen(
    H_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    hourly_labor_rate: float,
):
    # --- incremental liquefaction block --------------------------------
    base_capex_liq = CTX["tech"]["H2_liq"]["base_CapEx"]["value"]
    elect_liq = CTX["tech"]["H2_liq"]["elect_demand"]["value"]

    capex_liq = base_capex_liq * instal_factor
    opex_liq_components = {
        "Liquefaction Facility O&M": (CTX["glob"]["op_maint_rate"]["value"]
                                        + CTX["glob"]["tax_rate"]["value"]
                                        ) * capex_liq,
        "Liquefaction Facility Electricity": elect_liq * elect_price,
    }
    emiss_liq = elect_liq * elect_emissions_intensity

    # --- underlying STP hydrogen via generic engine --------------------
    prices = {
        "water": water_price,
        "ng": NG_price,
        "lcb": LCB_price,
        "elec": elect_price,
        "labor": hourly_labor_rate,
    }
    capex_h2, opex_h2, emiss_h2, capex_h2_components, opex_h2_components, emiss_h2_components = generic_production(
        H_pathway,
        "H2",
        instal_factor,
        prices,
        elect_emissions_intensity,
        lcb_upstream_emiss=LCB_upstream_emissions
    )
    
    capex_components = capex_h2_components
    capex_components["Liquefaction Facility"] = capex_liq

    opex_components = {**opex_liq_components, **opex_h2_components}

    emiss_components = emiss_h2_components
    emiss_components["Electricity for Liquefaction"] = emiss_liq

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components

# ────────────────────────────────────────────────────────────────────
# Fossil natural gas at STP
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_NG(
    water_price, NG_price, elect_price, elect_emissions_intensity
):
    prices = {
        "water": water_price,
        "ng": NG_price,
        "lcb": 0.0,
        "elec": elect_price,
        "labor": 0.0,
    }
    return generic_production(
        "NG",  # canonical table key
        "NG",
        install_factor=1.0,
        prices=prices,
        elec_intensity=elect_emissions_intensity
    )

# ────────────────────────────────────────────────────────────────────
# Liquefied fossil natural gas
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_liquid_NG(
    instal_factor,
    water_price,
    NG_price,
    elect_price,
        elect_emissions_intensity,
):
    """Incremental liquefaction block + generic NG feedstock."""
    # incremental liquefaction
    capex_liq = (
        CTX["tech"]["NG_liq"]["base_CapEx_2018"]["value"]
        * CTX["glob"]["2018_to_2024_USD"]["value"]
        * instal_factor
    )
    opex_liq_components = {
        "Liquefaction Facility O&M": (CTX["glob"]["op_maint_rate"]["value"] + CTX["glob"]["tax_rate"]["value"])
        * capex_liq,
        "NG for Liquefaction": CTX["derived"]["NG_liq_NG_demand_GJ"] * NG_price,
        "Water for Liquefaction": CTX["derived"]["NG_liq_water_demand"] * water_price,
        "Electricity for Liquefaction": CTX["derived"]["NG_liq_elect_demand"] * elect_price
    }
    
    emiss_liq_components = {
        "Onsite CO2 Emissions for Liquefaction": CTX["derived"]["NG_liq_CO2_emissions"],
        "CH4 Leakage for Liquefaction": CTX["derived"]["NG_liq_CH4_leakage"] * CTX["glob"]["NG_GWP"]["value"],
        "Electricity Demand for Liquefaction": CTX["derived"]["NG_liq_elect_demand"] * elect_emissions_intensity
    }

    # upstream NG via generic engine
    capex_ng, opex_ng, emiss_ng, capex_ng_components, opex_ng_components, emiss_ng_components = calculate_production_costs_emissions_NG(
        water_price, NG_price, elect_price, elect_emissions_intensity
    )
    
    capex_components = capex_ng_components
    capex_components["Liquefaction facility"] = capex_liq
    
    opex_components = {**opex_liq_components, **opex_ng_components}
    
    emiss_components = {**emiss_liq_components, **emiss_ng_components}

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components
    
# ────────────────────────────────────────────────────────────────────
# Synthetic natural gas at STP
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_SNG(
    H_pathway: str,
    C_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    labor_rate: float,
):
    # --- core SNG block (no feeds baked in) ---------------------------
    elect = CTX["tech"]["SNG"]["elect_demand"]["value"]
    water = CTX["tech"]["SNG"]["water_demand"]["value"]
    ng = 0.0
    employees = CTX["tech"]["SNG"]["employees"]["value"]
    yearly_output = CTX["tech"]["SNG"]["yearly_output"]["value"]

    capex_sng = CTX["tech"]["SNG"]["base_CapEx_2016"]["value"] * CTX["glob"]["2016_to_2024_USD"]["value"] * instal_factor

    opex_sng_components = {
        "SNG Production Labor": CTX["glob"]["workhours_per_year"]["value"] * labor_rate
                                 * employees / yearly_output
                                 * (1 + CTX["glob"]["gen_admin_rate"]["value"]),
        "SNG Production O&M": (CTX["glob"]["op_maint_rate"]["value"] + CTX["glob"]["tax_rate"]["value"]) * capex_sng,
        "Electricity for SNG Production": elect * elect_price,
        "Water for SNG Production": water * water_price,
        "NG for SNG Production": ng * NG_price
    }
    emiss_sng_components = {
        "Electricity for SNG Production": elect * elect_emissions_intensity,
        "Onsite Emissions for SNG Production": CTX["tech"]["SNG"]["onsite_emissions"]["value"]
    }

    # --- H₂ feed (generic engine) ------------------------------------
    H2_req = CTX["tech"]["SNG"]["H2_demand"]["value"]
    prices = {"water": water_price, "ng": NG_price, "lcb": LCB_price, "elec": elect_price, "labor": labor_rate}
    capex_h2, opex_h2, emiss_h2, capex_h2c, opex_h2c, emiss_h2c = generic_production(
        H_pathway, "H2", instal_factor, prices, elect_emissions_intensity, lcb_upstream_emiss=LCB_upstream_emissions
    )
    capex_h2c = {k: v * H2_req for k, v in capex_h2c.items()}
    opex_h2c  = {k: v * H2_req for k, v in opex_h2c.items()}
    emiss_h2c = {k: v * H2_req for k, v in emiss_h2c.items()}

    # --- CO₂ feed (credit-aware helper) -------------------------------
    CO2_req = CTX["tech"]["SNG"]["CO2_demand"]["value"]
    credit_sng = CTX["molecular_info"]["MW_CO2"]["value"] / CTX["molecular_info"]["MW_CH4"]["value"]
    capex_c, opex_c, emiss_c, capex_cc, opex_cc, emiss_cc = _feed_co2(
        C_pathway, credit_sng, prices, elect_emissions_intensity, CO2_req
    )
    capex_cc = {k: v * CO2_req for k, v in capex_cc.items()}
    opex_cc  = {k: v * CO2_req for k, v in opex_cc.items()}

    # --- combine -------------------------------------------------------
    capex_components = {**capex_h2c, **capex_cc}
    capex_components["SNG Production Facility"] = capex_sng
    opex_components = {**opex_h2c, **opex_cc, **opex_sng_components}
    emiss_components = {**emiss_h2c, **emiss_cc, **emiss_sng_components}

    return (sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()),
            capex_components, opex_components, emiss_components)

# ────────────────────────────────────────────────────────────────────
# Liquefied synthetic natural gas at atmospheric pressure
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_LSNG(
    H_pathway: str,
    C_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    labor_rate: float,
):
    # --- incremental liquefaction block (reuse your LNG pattern) ------
    capex_liq = (
        CTX["tech"]["NG_liq"]["base_CapEx_2018"]["value"]
        * CTX["glob"]["2018_to_2024_USD"]["value"]   # inflate to 2024$
        * instal_factor
    )
    opex_liq_components = {
        "Liquefaction Facility O&M": (CTX["glob"]["op_maint_rate"]["value"] + CTX["glob"]["tax_rate"]["value"]) * capex_liq,
        "NG for Liquefaction": CTX["derived"]["NG_liq_NG_demand_GJ"] * NG_price,
        "Water for Liquefaction": CTX["derived"]["NG_liq_water_demand"] * water_price,
        "Electricity for Liquefaction": CTX["derived"]["NG_liq_elect_demand"] * elect_price,
    }
    emiss_liq_components = {
        "Onsite CO2 Emissions for Liquefaction": CTX["derived"]["NG_liq_CO2_emissions"],
        "CH4 Leakage for Liquefaction": CTX["derived"]["NG_liq_CH4_leakage"] * CTX["glob"]["NG_GWP"]["value"],
        "Electricity Demand for Liquefaction": CTX["derived"]["NG_liq_elect_demand"] * elect_emissions_intensity,
    }

    # --- underlying SNG (core) ---------------------------------------
    capex_sng, opex_sng, emiss_sng, capex_sng_c, opex_sng_c, emiss_sng_c = calculate_production_costs_emissions_SNG(
        H_pathway, C_pathway, instal_factor, water_price, NG_price, LCB_price,
        LCB_upstream_emissions, elect_price, elect_emissions_intensity, labor_rate
    )

    # totals (increment over SNG)
    capex_components  = {"Liquefaction Facility": capex_liq}
    opex_components   = {**opex_liq_components}
    emiss_components  = {**emiss_liq_components}

    return (sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()),
            capex_components, opex_components, emiss_components)

# ────────────────────────────────────────────────────────────────────
# Compressed gaseous H₂ at 700 bar
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_compressed_hydrogen(
    H_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    hourly_labor_rate: float,
):
    # --- incremental compression block ---------------------------------
    base_capex_comp = CTX["tech"]["H2_comp"]["base_CapEx"]["value"]
    elect_comp = CTX["tech"]["H2_comp"]["elect_demand"]["value"]

    capex_comp = base_capex_comp * instal_factor
    opex_comp_components = {
        "Compression Facility O&M": (CTX["glob"]["op_maint_rate"]["value"]
                                        + CTX["glob"]["tax_rate"]["value"]
                                        ) * capex_comp,
        "Electricity for Compression": elect_comp * elect_price
    }
    emiss_comp = elect_comp * elect_emissions_intensity

    # --- underlying STP hydrogen via generic engine --------------------
    prices = {
        "water": water_price,
        "ng": NG_price,
        "lcb": LCB_price,
        "elec": elect_price,
        "labor": hourly_labor_rate,
    }
    capex_h2, opex_h2, emiss_h2, capex_h2_components, opex_h2_components, emiss_h2_components = generic_production(
        H_pathway,
        "H2",
        instal_factor,
        prices,
        elect_emissions_intensity,
        LCB_upstream_emissions
    )
    
    capex_components = capex_h2_components
    capex_components["Compression Facility"] = capex_comp

    opex_components = {**opex_comp_components, **opex_h2_components}

    emiss_components = emiss_h2_components
    emiss_components["Electricity for Compression"] = emiss_comp

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components


# ────────────────────────────────────────────────────────────────────
# Liquid NH₃ (cryogenic, 1 bar) – flexible H₂ feed
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_ammonia(
    H_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    labor_rate: float,
):
    # --- core Haber-Bosch block (no feeds) ----------------------------
    elect = CTX["derived"]["NH3_elect_demand"]  # kWh / kg NH₃
    water = CTX["derived"]["NH3_water_demand"]  # m³ / kg NH₃
    ng = CTX["tech"]["NH3"]["NG_demand"]["value"]  # GJ / kg NH₃

    employees = CTX["derived"]["NH3_employees"]
    yearly_output = CTX["tech"]["NH3"]["yearly_output"]["value"]

    capex_nh3 = CTX["derived"]["NH3_base_CapEx"] * instal_factor
        
    opex_nh3_components = {
        "NH3 Production Labor": CTX["glob"]["workhours_per_year"]["value"]
                                    * labor_rate
                                    * employees
                                    / yearly_output
                                    * (1 + CTX["glob"]["gen_admin_rate"]["value"]),
        "NH3 Production O&M": (CTX["glob"]["op_maint_rate"]["value"] + CTX["glob"]["tax_rate"]["value"])
                                    * capex_nh3,
        "Electricity for NH3 Production": elect * elect_price,
        "Water for NH3 Production": water * water_price,
        "NG for NH3 Production": ng * NG_price
    }

    emiss_nh3_components = {
        "Electricity for NH3 Production": elect * elect_emissions_intensity,
        "NG for NH3 Production": ng / CTX["glob"]["NG_HHV"]["value"]    # fugitive CH4
        * CTX["glob"]["NG_GWP"]["value"]
        * CTX["derived"]["NG_CH4_leakage"]
        + ng / CTX["glob"]["NG_HHV"]["value"]
        * CTX["derived"]["NG_CO2_emissions"]  # upstream CO2 from NG production
    }

    # (No onsite-process CO₂ for NH₃, so nothing else to add)

    # --- H₂ feed (generic engine) ------------------------------------
    H2_demand = CTX["derived"]["NH3_H2_demand"]  # kg H₂ per kg NH₃

    prices = {
        "water": water_price,
        "ng": NG_price,
        "lcb": LCB_price,
        "elec": elect_price,
        "labor": labor_rate,
    }

    capex_h2, opex_h2, emiss_h2, capex_h2_components, opex_h2_components, emiss_h2_components = generic_production(
        H_pathway,
        "H2",
        instal_factor,
        prices,
        elect_emissions_intensity,
        LCB_upstream_emissions
    )
    
    # Scale all H2 components by the H2 demand of the NH3 production process
    capex_h2_components_scaled = {k: v * H2_demand for k, v in capex_h2_components.items()}
    opex_h2_components_scaled = {k: v * H2_demand for k, v in opex_h2_components.items()}
    emiss_h2_components_scaled = {k: v * H2_demand for k, v in emiss_h2_components.items()}
    
    # Combine all component dictionaries together
    capex_components = capex_h2_components_scaled
    capex_components["NH3 Production Facility"] = capex_nh3
    opex_components = {**opex_h2_components_scaled, **opex_nh3_components}
    emiss_components = {**emiss_h2_components_scaled, **emiss_nh3_components}

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components

def calculate_production_costs_emissions_methanol(
    H_pathway: str,
    C_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    labor_rate: float,
):
    # ── “core” MeOH synthesis block (no feeds) ────────────────────────
    elect = CTX["tech"]["MeOH"]["elect_demand"]["value"]
    water = CTX["tech"]["MeOH"]["water_demand"]["value"]
    ng = CTX["tech"]["MeOH"]["NG_demand"]["value"]
    employees = CTX["tech"]["MeOH"]["employees"]["value"]
    yearly_output = CTX["tech"]["MeOH"]["yearly_output"]["value"]

    capex_methanol = CTX["tech"]["MeOH"]["base_CapEx"]["value"] * instal_factor
    
    opex_methanol_components = {
        "Methanol Production Labor": CTX["glob"]["workhours_per_year"]["value"]
                                        * labor_rate
                                        * employees
                                        / yearly_output
                                        * (1 + CTX["glob"]["gen_admin_rate"]["value"]),
        "Methanol Production O&M": (CTX["glob"]["op_maint_rate"]["value"] + CTX["glob"]["tax_rate"]["value"])
                                * capex_methanol,
        "Electricity for Methanol Production": elect * elect_price,
        "Water for Methanol Production": water * water_price,
        "NG for Methanol Production": ng * NG_price
    }

    emiss_methanol_components = {
        "Electricity for Methanol Production": elect * elect_emissions_intensity,
        "NG for Methanol Production": ng / CTX["glob"]["NG_HHV"]["value"]    # fugitive CH4
                                        * CTX["glob"]["NG_GWP"]["value"]
                                        * CTX["derived"]["NG_CH4_leakage"]
                                        + ng / CTX["glob"]["NG_HHV"]["value"]
                                        * CTX["derived"]["NG_CO2_emissions"]  # upstream CO2 from NG production
    }

    if C_pathway not in ("BEC", "DAC"):  # keep legacy rule
        emiss_methanol_components["Onsite Emissions for Methanol Production"] = CTX["tech"]["MeOH"]["onsite_emissions"]["value"]

    # ── H₂ and CO₂ feed handling via helpers ──────────────────────────
    H2_demand = CTX["tech"]["MeOH"]["H2_demand"]["value"]
    CO2_demand = CTX["tech"]["MeOH"]["CO2_demand"]["value"]

    prices = {
        "water": water_price,
        "ng": NG_price,
        "lcb": LCB_price,
        "elec": elect_price,
        "labor": labor_rate,
    }

    capex_h2, opex_h2, emiss_h2, capex_h2_components, opex_h2_components, emiss_h2_components = generic_production(
        H_pathway,
        "H2",
        instal_factor,
        prices,
        elect_emissions_intensity,
        lcb_upstream_emiss=LCB_upstream_emissions
    )
    
    # Scale all H2 components by the H2 demand of the methanol production process
    capex_h2_components_scaled = {k: v * H2_demand for k, v in capex_h2_components.items()}
    opex_h2_components_scaled = {k: v * H2_demand for k, v in opex_h2_components.items()}
    emiss_h2_components_scaled = {k: v * H2_demand for k, v in emiss_h2_components.items()}

    credit_methanol = (
        CTX["molecular_info"]["MW_CO2"]["value"]
        / CTX["molecular_info"]["MW_MeOH"]["value"]
    )
    capex_co2, opex_co2, emiss_co2, capex_co2_components, opex_co2_components, emiss_co2_components = _feed_co2(
        C_pathway, credit_methanol, prices, elect_emissions_intensity, CO2_demand
    )
    
    # Scale the CapEx and OpEx CO2 components by the CO2 demand of the methanol production process (the emissions components are already scaled)
    capex_co2_components_scaled = {k: v * CO2_demand for k, v in capex_co2_components.items()}
    opex_co2_components_scaled = {k: v * CO2_demand for k, v in opex_co2_components.items()}

    # Combine all component dictionaries together
    capex_components = {**capex_h2_components_scaled, **capex_co2_components_scaled}
    capex_components["Methanol Production Facility"] = capex_methanol
    opex_components = {**opex_h2_components_scaled, **opex_co2_components_scaled, **opex_methanol_components}
    emiss_components = {**emiss_h2_components_scaled, **emiss_co2_components, **emiss_methanol_components}


    # ── totals (per kg MeOH) ──────────────────────────────────────────
#    CapEx = cap_core + cap_h2 * H2_demand + cap_co2 * CO2_demand
#    OpEx = op_core + op_h2 * H2_demand + op_co2 * CO2_demand
#    Emiss = em_core + em_h2 * H2_demand + em_co2

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components


# -----------------------------------------------------------------------
# Fischer–Tropsch diesel  (CxHyOz)  –  flexible H₂ & CO₂ feeds
# -----------------------------------------------------------------------
def calculate_production_costs_emissions_FTdiesel(
    H_pathway: str,
    C_pathway: str,
    instal_factor: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_upstream_emissions: float,
    elect_price: float,
    elect_emissions_intensity: float,
    labor_rate: float,
):
    # ── “core” FT block (no feeds) ────────────────────────────────────
    elect = CTX["tech"]["FTdiesel"]["elect_demand"]["value"]
    water = CTX["tech"]["FTdiesel"]["water_demand"]["value"]
    ng = CTX["tech"]["FTdiesel"]["NG_demand"]["value"]
    employees = CTX["tech"]["FTdiesel"]["employees"]["value"]
    yearly_output = CTX["tech"]["FTdiesel"]["yearly_output"]["value"]

    capex_ftdiesel = CTX["tech"]["FTdiesel"]["base_CapEx"]["value"] * instal_factor

    opex_ftdiesel_components = {
        "FT Diesel Production Labor": CTX["glob"]["workhours_per_year"]["value"]
                                        * labor_rate
                                        * employees
                                        / yearly_output
                                        * (1 + CTX["glob"]["gen_admin_rate"]["value"]),
        "FT Diesel Production O&M": (CTX["glob"]["op_maint_rate"]["value"] + CTX["glob"]["tax_rate"]["value"])
                                * capex_ftdiesel,
        "Electricity for FT Diesel Production": elect * elect_price,
        "Water for FT Diesel Production": water * water_price,
        "NG for FT Diesel Production": ng * NG_price
    }

    emiss_ftdiesel_components = {
        "Electricity for FT Diesel Production": elect * elect_emissions_intensity,
        "NG for FT Diesel Production": ng / CTX["glob"]["NG_HHV"]["value"]    # fugitive CH4
                                        * CTX["glob"]["NG_GWP"]["value"]
                                        * CTX["derived"]["NG_CH4_leakage"]
                                        + ng / CTX["glob"]["NG_HHV"]["value"]
                                        * CTX["derived"]["NG_CO2_emissions"]  # upstream CO2 from NG production
    }

    if C_pathway not in ("BEC", "DAC"):
        emiss_ftdiesel_components["Onsite Emissions for FT Diesel Production"] = CTX["tech"]["FTdiesel"]["onsite_emissions"]["value"]

    # ── H₂ and CO₂ feeds ──────────────────────────────────────────────
    H2_demand = CTX["tech"]["FTdiesel"]["H2_demand"]["value"]
    CO2_demand = CTX["tech"]["FTdiesel"]["CO2_demand"]["value"]

    prices = {
        "water": water_price,
        "ng": NG_price,
        "lcb": LCB_price,
        "elec": elect_price,
        "labor": labor_rate,
    }

    capex_h2, opex_h2, emiss_h2, capex_h2_components, opex_h2_components, emiss_h2_components = generic_production(
        H_pathway,
        "H2",
        instal_factor,
        prices,
        elect_emissions_intensity,
        lcb_upstream_emiss=LCB_upstream_emissions
    )

    # Scale all H2 components by the H2 demand of the FT diesel production process
    capex_h2_components_scaled = {k: v * H2_demand for k, v in capex_h2_components.items()}
    opex_h2_components_scaled = {k: v * H2_demand for k, v in opex_h2_components.items()}
    emiss_h2_components_scaled = {k: v * H2_demand for k, v in emiss_h2_components.items()}

    # fuel-specific credit factor:  nC * MW_CO2  /  MW_FTdiesel
    credit_ft = (
        CTX["molecular_info"]["nC_FTdiesel"]["value"]
        * CTX["molecular_info"]["MW_CO2"]["value"]
        / CTX["molecular_info"]["MW_FTdiesel"]["value"]
    )

    capex_co2, opex_co2, emiss_co2, capex_co2_components, opex_co2_components, emiss_co2_components = _feed_co2(
        C_pathway, credit_ft, prices, elect_emissions_intensity, CO2_demand
    )
    
    # Scale the CapEx and OpEx CO2 components by the CO2 demand of the methanol production process (the emissions components are already scaled)
    capex_co2_components_scaled = {k: v * CO2_demand for k, v in capex_co2_components.items()}
    opex_co2_components_scaled = {k: v * CO2_demand for k, v in opex_co2_components.items()}
    
    # Combine all component dictionaries together
    capex_components = {**capex_h2_components_scaled, **capex_co2_components_scaled}
    capex_components["FT Diesel Production Facility"] = capex_ftdiesel
    opex_components = {**opex_h2_components_scaled, **opex_co2_components_scaled, **opex_ftdiesel_components}
    emiss_components = {**emiss_h2_components_scaled, **emiss_co2_components, **emiss_ftdiesel_components}

#    # ── totals (per kg FT-diesel) ─────────────────────────────────────
#    CapEx = cap_core + cap_h2 * H2_demand + cap_co2 * CO2_demand
#    OpEx = op_core + op_h2 * H2_demand + op_co2 * CO2_demand
#    Emiss = em_core + em_h2 * H2_demand + em_co2

    return sum(capex_components.values()), sum(opex_components.values()), sum(emiss_components.values()), capex_components, opex_components, emiss_components


def calculate_resource_demands_STP_hydrogen(H_pathway: str):
    """Return electricity, LCB, NG, water, CO₂ per kg H₂ for *H_pathway*."""
    elect = _p(H_pathway, "elect_demand")
    lcb = _p(H_pathway, "lcb_demand")
    ng = _p(H_pathway, "ng_demand")
    water = _p(H_pathway, "water_demand")
    return elect, lcb, ng, water, 0.0  # STP hydrogen never consumes external CO₂


def calculate_resource_demands_liquid_hydrogen(H_pathway):
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = (
        calculate_resource_demands_STP_hydrogen(H_pathway)
    )
    elect_demand += CTX["tech"]["H2_liq"]["elect_demand"]["value"]

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_compressed_hydrogen(H_pathway):
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = (
        calculate_resource_demands_STP_hydrogen(H_pathway)
    )
    elect_demand += CTX["tech"]["H2_comp"]["elect_demand"]["value"]

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_NG():
    water_demand = CTX["derived"]["NG_water_demand"]  # m^3 water / kg NG
    NG_demand = (
        CTX["glob"]["NG_HHV"]["value"] + CTX["derived"]["NG_NG_demand_GJ"]
    )  # GJ NG / kg NG
    elect_demand = CTX["derived"]["NG_elect_demand"]
    LCB_demand = 0
    CO2_demand = 0

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_liquid_NG():
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = (
        calculate_resource_demands_NG()
    )
    NG_demand += CTX["derived"]["NG_liq_NG_demand_GJ"]
    water_demand += CTX["derived"]["NG_liq_water_demand"]
    elect_demand += CTX["derived"]["NG_liq_elect_demand"]

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand
    
def calculate_resource_demands_SNG(H_pathway, C_pathway):
    elect = CTX["tech"]["SNG"]["elect_demand"]["value"]
    water = CTX["tech"]["SNG"]["water_demand"]["value"]
    ng = 0.0
    lcb = 0.0
    co2 = CTX["tech"]["SNG"]["CO2_demand"]["value"]

    # Add H2 pathway demands (scaled by SNG H2 demand)
    H2_e, H2_lcb, H2_ng, H2_w, H2_co2 = calculate_resource_demands_STP_hydrogen(H_pathway)
    H2_req = CTX["tech"]["SNG"]["H2_demand"]["value"]
    elect += H2_e * H2_req
    lcb += H2_lcb * H2_req
    ng += H2_ng * H2_req
    water += H2_w * H2_req

    # If DAC is the C pathway, add DAC upstream burdens (per kg CO2)
    if C_pathway == "DAC":
        ng += CTX["derived"]["DAC_CO2_upstream_NG"] * co2
        water += CTX["derived"]["DAC_CO2_upstream_water"] * co2
        elect += CTX["derived"]["DAC_CO2_upstream_elect"] * co2

    return elect, lcb, ng, water, co2


def calculate_resource_demands_LSNG(H_pathway, C_pathway):
    elect, lcb, ng, water, co2 = calculate_resource_demands_SNG(H_pathway, C_pathway)
    elect += CTX["derived"]["NG_liq_elect_demand"]
    ng    += CTX["derived"]["NG_liq_NG_demand_GJ"]
    water += CTX["derived"]["NG_liq_water_demand"]
    return elect, lcb, ng, water, co2



def calculate_resource_demands_ammonia(H_pathway):
    elect_demand = CTX["derived"]["NH3_elect_demand"]
    LCB_demand = 0
    H2_demand = CTX["derived"]["NH3_H2_demand"]
    CO2_demand = 0
    NG_demand = CTX["tech"]["NH3"]["NG_demand"]["value"]
    water_demand = CTX["derived"]["NH3_water_demand"]

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = (
        calculate_resource_demands_STP_hydrogen(H_pathway)
    )
    elect_demand += H2_elect_demand * H2_demand
    LCB_demand += H2_LCB_demand * H2_demand
    NG_demand += H2_NG_demand * H2_demand
    water_demand += H2_water_demand * H2_demand
    CO2_demand += H2_CO2_demand * H2_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_methanol(H_pathway, C_pathway):
    elect_demand = CTX["tech"]["MeOH"]["elect_demand"]["value"]
    LCB_demand = CTX["tech"]["MeOH"]["LCB_demand"]["value"]
    H2_demand = CTX["tech"]["MeOH"]["H2_demand"]["value"]
    CO2_demand = CTX["tech"]["MeOH"]["CO2_demand"]["value"]
    NG_demand = CTX["tech"]["MeOH"]["NG_demand"]["value"]
    water_demand = CTX["tech"]["MeOH"]["water_demand"]["value"]

    if C_pathway == "DAC":
        NG_demand = NG_demand + CTX["derived"]["DAC_CO2_upstream_NG"] * CO2_demand
        water_demand = (
            water_demand + CTX["derived"]["DAC_CO2_upstream_water"] * CO2_demand
        )
        elect_demand = (
            elect_demand + CTX["derived"]["DAC_CO2_upstream_elect"] * CO2_demand
        )

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = (
        calculate_resource_demands_STP_hydrogen(H_pathway)
    )
    elect_demand += H2_elect_demand * H2_demand
    LCB_demand += H2_LCB_demand * H2_demand
    NG_demand += H2_NG_demand * H2_demand
    water_demand += H2_water_demand * H2_demand
    CO2_demand += H2_CO2_demand * H2_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_FTdiesel(H_pathway, C_pathway):
    elect_demand = CTX["tech"]["FTdiesel"]["elect_demand"]["value"]
    LCB_demand = CTX["tech"]["FTdiesel"]["LCB_demand"]["value"]
    H2_demand = CTX["tech"]["FTdiesel"]["H2_demand"]["value"]
    CO2_demand = CTX["tech"]["FTdiesel"]["CO2_demand"]["value"]
    NG_demand = CTX["tech"]["FTdiesel"]["NG_demand"]["value"]
    water_demand = CTX["tech"]["FTdiesel"]["water_demand"]["value"]

    if C_pathway == "DAC":
        NG_demand = NG_demand + CTX["derived"]["DAC_CO2_upstream_NG"] * CO2_demand
        water_demand = (
            water_demand + CTX["derived"]["DAC_CO2_upstream_water"] * CO2_demand
        )
        elect_demand = (
            elect_demand + CTX["derived"]["DAC_CO2_upstream_elect"] * CO2_demand
        )

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = (
        calculate_resource_demands_STP_hydrogen(H_pathway)
    )
    elect_demand += H2_elect_demand * H2_demand
    LCB_demand += H2_LCB_demand * H2_demand
    NG_demand += H2_NG_demand * H2_demand
    water_demand += H2_water_demand * H2_demand
    CO2_demand += H2_CO2_demand * H2_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


# Dispatch table for resource demands
resource_demand_fn = {
    "hydrogen": calculate_resource_demands_STP_hydrogen,
    "liquid_hydrogen": calculate_resource_demands_liquid_hydrogen,
    "compressed_hydrogen": calculate_resource_demands_compressed_hydrogen,
    "ammonia": calculate_resource_demands_ammonia,
    "methanol": calculate_resource_demands_methanol,
    "FTdiesel": calculate_resource_demands_FTdiesel,
    "ng": calculate_resource_demands_NG,
    "lng": calculate_resource_demands_liquid_NG,
    "sng":  calculate_resource_demands_SNG,
    "lsng": calculate_resource_demands_LSNG,
}

# Dispatch table for production cost + emissions
cost_emission_fn = {
    "hydrogen": calculate_production_costs_emissions_STP_hydrogen,
    "liquid_hydrogen": calculate_production_costs_emissions_liquid_hydrogen,
    "compressed_hydrogen": calculate_production_costs_emissions_compressed_hydrogen,
    "ammonia": calculate_production_costs_emissions_ammonia,
    "methanol": calculate_production_costs_emissions_methanol,
    "FTdiesel": calculate_production_costs_emissions_FTdiesel,
    "ng": calculate_production_costs_emissions_NG,
    "lng": calculate_production_costs_emissions_liquid_NG,
    "sng": calculate_production_costs_emissions_SNG,
    "lsng": calculate_production_costs_emissions_LSNG,
}

fuel_comments = {
    "hydrogen": "hydrogen at standard temperature and pressure",
    "liquid_hydrogen": "Liquid cryogenic hydrogen at atmospheric pressure",
    "compressed_hydrogen": "compressed gaseous hydrogen at 700 bar",
    "ammonia": "liquid cryogenic ammonia at atmospheric pressure",
    "methanol": "liquid methanol at STP",
    "FTdiesel": "liquid Fischer--Tropsch diesel fuel at STP",
    "ng": "natural gas at standard temperature and pressure",
    "lng": "liquid natural gas at atmospheric pressure",
    "sng": "synthetic natural gas at standard temperature and pressure",
    "lsng": "liquid synthetic natural gas at atmospheric pressure",
}

def load_projection_if_year_given(input_dir, file_prefix, region, year, unit_label):
    """
    If year is provided, load the projection file for the given prefix (e.g., 'grid_electricity_price')
    and return the value for the specified region and year. If not, return None.
    """
    if year is None:
        return None
    filename = os.path.join(input_dir, f"{file_prefix}_projection.csv")
    df = pd.read_csv(filename).set_index("Region")
    col = f"{year} [{unit_label}]"
    return df.loc[region, col]

def main(save_breakdowns=False, year=None, include_demands=False):
    input_dir = f"{top_dir}/input_fuel_pathway_data/"
    output_dir_production = f"{top_dir}/input_fuel_pathway_data/production/"
    output_year = year if year is not None else 2024
    ensure_directory_exists(output_dir_production)
    output_dir_production_components = os.path.join(output_dir_production, "cost_emissions_components")
    if save_breakdowns:
        ensure_directory_exists(output_dir_production_components)
        
    output_dir_process = f"{top_dir}/input_fuel_pathway_data/process/"
    ensure_directory_exists(output_dir_process)
    output_dir_process_components = os.path.join(output_dir_process, "cost_emissions_components")
    if save_breakdowns:
        ensure_directory_exists(output_dir_process_components)

    # Read the input CSV files
    input_df = pd.read_csv(input_dir + "regional_TEA_inputs.csv")
    pathway_df = pd.read_csv(input_dir + "fuel_pathway_options.csv")

    # Populate the arrays using the columns
    fuels_and_contents = pathway_df[["WTG fuels", "fuel contents"]].dropna()
    fuels = fuels_and_contents["WTG fuels"].tolist()
    fuel_content_map = dict(
        zip(fuels_and_contents["WTG fuels"], fuels_and_contents["fuel contents"])
    )
    processes = pathway_df["GTT processes"].dropna().tolist()

    def get_unique_sources(column):
        return sorted(
            set(
                source.strip()
                for sources in pathway_df[column].dropna()
                for source in str(sources).split(",")
                if source.strip()
            )
        )

    Esources = get_unique_sources("electricity sources")
    Hsources = get_unique_sources("hydrogen sources")
    Csources = get_unique_sources("carbon sources")
    # Well to Gate fuel production
    for fuel in fuels:
        if "fossil" in fuel_content_map[fuel]:
            fuel_pathways = ["fossil"]
            H_pathways = ["n/a"]
            C_pathways = ["n/a"]
            E_pathways = ["n/a"]
        else:
            if "C" in fuel_content_map[fuel]:  # if fuel contains carbon
                fuel_pathways_noelec = []
                H_pathways_noelec = []
                C_pathways_noelec = []
                for Csource in Csources:
                    for Hsource in Hsources:
                        if Hsource == Csource:
                            H_pathways_noelec += [Hsource]
                            C_pathways_noelec += [Csource]
                            fuel_pathways_noelec += [Hsource + "_H_C"]
                        elif (
                            (
                                (Hsource == "SMR")
                                & ((Csource == "SMRCCS") | (Csource == "ATRCCS"))
                            )
                            | ((Hsource != "SMR") & (Csource == "SMR"))
                            | ((Hsource != "BG") & (Csource == "BG"))
                        ):
                            # skip case where ATRCCS/SMRCCS is used for C and SMR is used for H, because this does not make sense.
                            # also skip cases where BG or SMR is used for C but not H, because C would not be captured or usable in those cases.
                            continue
                        else:
                            H_pathways_noelec += [Hsource]
                            C_pathways_noelec += [Csource]
                            fuel_pathways_noelec += [Hsource + "_H_" + Csource + "_C"]
            else:  # fuel does not contain carbon
                fuel_pathways_noelec = [Hsource + "_H" for Hsource in Hsources]
                H_pathways_noelec = [Hsource for Hsource in Hsources]
                C_pathways_noelec = ["n/a" for Hsource in Hsources]
            fuel_pathways = []
            H_pathways = []
            C_pathways = []
            E_pathways = []
            for Esource in Esources:
                H_pathways += [
                    H_pathway_noelec for H_pathway_noelec in H_pathways_noelec
                ]
                C_pathways += [
                    C_pathway_noelec for C_pathway_noelec in C_pathways_noelec
                ]
                E_pathways += [Esource for fuel_pathway_noelec in fuel_pathways_noelec]
                fuel_pathways += [
                    fuel_pathway_noelec + "_" + Esource + "_E"
                    for fuel_pathway_noelec in fuel_pathways_noelec
                ]

        # List to hold all rows for the output CSV
        output_data = []
        # GE - list to hold all resource rows for the ouput csv
        output_resource_data = []

        # Iterate through each row in the input data and perform calculations
        for fuel_pathway in fuel_pathways:
            pathway_index = fuel_pathways.index(fuel_pathway)
            H_pathway = H_pathways[pathway_index]
            C_pathway = C_pathways[pathway_index]


            if include_demands:
                # Get resource demand function and call it
                if fuel in resource_demand_fn:
                    if fuel in ["methanol", "FTdiesel", "sng", "lsng"]:
                        elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = (
                            resource_demand_fn[fuel](H_pathway, C_pathway)
                        )
                    elif fuel in ["ammonia"]:
                        elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = (
                            resource_demand_fn[fuel](H_pathway)
                        )
                    elif fuel in ["ng", "lng"]:
                        elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = (
                            resource_demand_fn[fuel]()
                        )
                    else:
                        elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = (
                            resource_demand_fn[fuel](H_pathway)
                        )

                calculated_resource_row = [
                    fuel,
                    H_pathway,
                    C_pathway,
                    fuel_pathway,
                    elect_demand,
                    LCB_demand,
                    NG_demand,
                    water_demand,
                    CO2_demand,
                ]
                output_resource_data.append(calculated_resource_row)

            for row_index, row in input_df.iterrows():
                (
                    region,
                    instal_factor_low,
                    instal_factor_high,
                    src,
                    water_price,
                    src,
                    NG_price,
                    src,
                    NG_fugitive_emissions,
                    src,
                    LCB_price,
                    src,
                    LCB_upstream_emissions,
                    src,
                    grid_price,
                    src,
                    grid_emissions_intensity,
                    src,
                    solar_price,
                    src,
                    solar_emissions_intensity,
                    src,
                    wind_price,
                    src,
                    wind_emissions_intensity,
                    src,
                    nuke_price,
                    src,
                    nuke_emissions_intensity,
                    src,
                    hourly_labor_rate,
                    src,
                ) = row

                # Calculate the average installation factor
                instal_factor = (instal_factor_low + instal_factor_high) / 2
                H_pathway = H_pathways[pathway_index]
                C_pathway = C_pathways[pathway_index]
                E_pathway = E_pathways[pathway_index]
                # Check whether we are working with grid or renewable electricity
                # if renewables, fix electricity price and emissions to imposed values
                if E_pathway == "grid":
                    proj_price = load_projection_if_year_given(input_dir, "grid_electricity_price", region, year, "2024$/kWh")
                    proj_emiss = load_projection_if_year_given(input_dir, "grid_electricity_emissions", region, year, "kgCO2e/kWh")
                    elect_price = proj_price if proj_price is not None else grid_price
                    elect_emissions_intensity = proj_emiss if proj_emiss is not None else grid_emissions_intensity
                elif E_pathway == "solar":
                    elect_price = solar_price
                    elect_emissions_intensity = solar_emissions_intensity
                elif E_pathway == "wind":
                    elect_price = wind_price
                    elect_emissions_intensity = wind_emissions_intensity
                elif E_pathway == "nuke":
                    elect_price = nuke_price
                    elect_emissions_intensity = nuke_emissions_intensity

                # Get the appropriate function for cost/emissions
                if fuel in cost_emission_fn:
                    if fuel in ["methanol", "FTdiesel", "sng", "lsng"]:
                        CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = cost_emission_fn[fuel](
                            H_pathway,
                            C_pathway,
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    elif fuel in ["ammonia"]:
                        CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = cost_emission_fn[fuel](
                            H_pathway,
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    elif fuel == "ng":
                        CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = cost_emission_fn[fuel](
                            water_price,
                            NG_price,
                            elect_price,
                            elect_emissions_intensity,
                        )
                    elif fuel == "lng":
                        CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = cost_emission_fn[fuel](
                            instal_factor,
                            water_price,
                            NG_price,
                            elect_price,
                            elect_emissions_intensity,
                        )
                    else:
                        CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = cost_emission_fn[fuel](
                            H_pathway,
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    # Retrieve the descriptive comment string
                    comment = fuel_comments.get(fuel, "")

                CapEx *= 1000  # convert to $/tonne
                CapEx_components = {k: v * 1000 for k, v in CapEx_components.items()}
                OpEx *= 1000  # convert to $/tonne
                OpEx_components = {k: v * 1000 for k, v in OpEx_components.items()}
                LCOF = CapEx + OpEx  # in $/tonne
                calculated_row = [
                    fuel,
                    H_pathway,
                    C_pathway,
                    E_pathway,
                    fuel_pathway,
                    region,
                    1,
                    output_year,
                    CapEx,
                    OpEx,
                    LCOF,
                    emissions,
                    comment,
                ]
                output_data.append(calculated_row)
                if save_breakdowns:
                    region_safe = region.replace(" ", "_").lower()
                    path_prefix = os.path.join(output_dir_production_components, f"{fuel}_{fuel_pathway}_{region_safe}")
                    
                    combined_components = {
                        "CapEx (2024$/tonne)": CapEx_components,
                        "OpEx (2024$/tonne)": OpEx_components,
                        "emissions (kg CO2e / kg fuel)": emissions_components
                    }
                    with open(f"{path_prefix}_components.json", "w") as f:
                        json.dump(combined_components, f, indent=2)

        # GE - Define the resource output to CSV column names - may need to add more columns
        output_resource_columns = [
            "Fuel",
            "Hydrogen Source",
            "Carbon Source",
            "Fuel Pathway",
            "Electricity Demand [kWh / kg fuel]",
            "Lignocellulosic Biomass Demand [kg / kg fuel]",
            "NG Demand [GJ / kg fuel]",
            "Water Demand [m^3 / kg fuel]",
            "CO2 Demand [kg CO2 / kg fuel]",
        ]

        # Create a DataFrame for the output data
        resource_df = pd.DataFrame(
            output_resource_data, columns=output_resource_columns
        )

        # Write the output data to a CSV file
        if include_demands:
            output_resource_file = f"{fuel}_resource_demands.csv"
            resource_df.to_csv(
                os.path.join(output_dir_production, output_resource_file), index=False
            )
            print(
                f"Output CSV file created: {os.path.join(output_dir_production, output_resource_file)}"
            )

        # Define the output CSV column names
        output_columns = [
            "Fuel",
            "Hydrogen Source",
            "Carbon Source",
            "Electricity Source",
            "Pathway Name",
            "Region",
            "Number",
            "Year",
            "CapEx [$/tonne]",
            "OpEx [$/tonne]",
            "LCOF [$/tonne]",
            "Emissions [kg CO2e / kg fuel]",
            "Comment",
        ]

        # Create a DataFrame for the output data
        output_df = pd.DataFrame(output_data, columns=output_columns)

        # Write the output data to a CSV file
        suffix = f"_{year}" if year is not None else ""
        output_file = f"{fuel}_costs_emissions{suffix}.csv"
        output_df.to_csv(os.path.join(output_dir_production, output_file), index=False)

        print(
            f"Output CSV file created: {os.path.join(output_dir_production, output_file)}"
        )

    # Gate to Pump Processes
    processes = sorted(
        set(
            proc.strip()
            for procs in pathway_df["GTT processes"].dropna()
            for proc in str(procs).split(",")
            if proc.strip()
        )
    )

    for process in processes:
        if process == "ng_liquefaction":
            process_pathways = ["fossil"]
            E_pathways = ["n/a"]
        else:
            process_pathways = Esources
            E_pathways = Esources

        # List to hold all rows for the output CSV
        output_data = []

        # Iterate through each row in the input data and perform calculations
        for process_pathway in process_pathways:
            pathway_index = process_pathways.index(process_pathway)
            for row_index, row in input_df.iterrows():
                (
                    region,
                    instal_factor_low,
                    instal_factor_high,
                    src,
                    water_price,
                    src,
                    NG_price,
                    src,
                    NG_fugitive_emissions,
                    src,
                    LCB_price,
                    src,
                    LCB_upstream_emissions,
                    src,
                    grid_price,
                    src,
                    grid_emissions_intensity,
                    src,
                    solar_price,
                    src,
                    solar_emissions_intensity,
                    src,
                    wind_price,
                    src,
                    wind_emissions_intensity,
                    src,
                    nuke_price,
                    src,
                    nuke_emissions_intensity,
                    src,
                    hourly_labor_rate,
                    src,
                ) = row
                # Take the average of the upper and lower installation cost factor
                instal_factor = (instal_factor_low + instal_factor_high) / 2
                H_pathway = "n/a"
                C_pathway = "n/a"
                E_pathway = E_pathways[pathway_index]
                # Check whether we are working with grid or renewable electricity
                if E_pathway == "grid":
                    elect_price = grid_price
                    elect_emissions_intensity = grid_emissions_intensity
                elif E_pathway == "solar":
                    elect_price = solar_price
                    elect_emissions_intensity = solar_emissions_intensity
                elif E_pathway == "wind":
                    elect_price = wind_price
                    elect_emissions_intensity = wind_emissions_intensity
                elif E_pathway == "nuke":
                    elect_price = nuke_price
                    elect_emissions_intensity = nuke_emissions_intensity
                # Calculations use LTE pathway for all, but subtract away the LTE costs/emissions
                if process == "hydrogen_liquefaction":
                    CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = (
                        calculate_production_costs_emissions_liquid_hydrogen(
                            "LTE",
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    )
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod, CapEx_H2prod_components, OpEx_H2prod_components, emissions_H2prod_components = (
                        calculate_production_costs_emissions_STP_hydrogen(
                            "LTE",
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    )
                    CapEx -= CapEx_H2prod
                    OpEx -= OpEx_H2prod
                    emissions -= emissions_H2prod
                    fuel = "liquid_hydrogen"
                    comment = "Liquefaction of STP hydrogen to cryogenic hydrogen at atmospheric pressure"
                elif process == "hydrogen_compression":
                    CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = (
                        calculate_production_costs_emissions_compressed_hydrogen(
                            "LTE",
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    )
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod, CapEx_H2prod_components, OpEx_H2prod_components, emissions_H2prod_components = (
                        calculate_production_costs_emissions_STP_hydrogen(
                            "LTE",
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    )
                    CapEx -= CapEx_H2prod
                    OpEx -= OpEx_H2prod
                    emissions -= emissions_H2prod
                    fuel = "compressed_hydrogen"
                    comment = (
                        "compression of STP hydrogen to gaseous hydrogen at 700 bar"
                    )
                elif process == "hydrogen_to_ammonia_conversion":
                    CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = (
                        calculate_production_costs_emissions_ammonia(
                            "LTE",
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    )
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod, CapEx_H2prod_components, OpEx_H2prod_components, emissions_H2prod_components = (
                        calculate_production_costs_emissions_STP_hydrogen(
                            "LTE",
                            instal_factor,
                            water_price,
                            NG_price,
                            LCB_price,
                            LCB_upstream_emissions,
                            elect_price,
                            elect_emissions_intensity,
                            hourly_labor_rate,
                        )
                    )
                    CapEx -= CapEx_H2prod * CTX["derived"]["NH3_H2_demand"]
                    OpEx -= OpEx_H2prod * CTX["derived"]["NH3_H2_demand"]
                    emissions -= emissions_H2prod * CTX["derived"]["NH3_H2_demand"]
                    fuel = "ammonia"
                    comment = "conversion of STP hydrogen to liquid cryogenic ammonia at atmospheric pressure"
                elif process == "ng_liquefaction":
                    CapEx, OpEx, emissions, CapEx_components, OpEx_components, emissions_components = (
                        calculate_production_costs_emissions_liquid_NG(
                            instal_factor,
                            water_price,
                            NG_price,
                            elect_price,
                            elect_emissions_intensity,
                        )
                    )
                    CapEx_NGprod, OpEx_NGprod, emissions_NGprod, CapEx_NGprod_components, OpEx_NGprod_components, emissions_NGprod_components = (
                        calculate_production_costs_emissions_NG(
                            water_price,
                            NG_price,
                            elect_price,
                            elect_emissions_intensity,
                        )
                    )
                    CapEx -= CapEx_NGprod
                    OpEx -= OpEx_NGprod
                    emissions -= emissions_NGprod
                    fuel = "ng"
                    comment = "conversion of STP natural gas to liquid cryogenic natural gas at atmospheric pressure"
                CapEx *= 1000  # convert to $/tonne
                OpEx *= 1000  # convert to $/tonne
                LCOF = CapEx + OpEx  # in $/tonne
                calculated_row = [
                    fuel,
                    H_pathway,
                    C_pathway,
                    E_pathway,
                    process_pathway,
                    region,
                    1,
                    output_year,
                    CapEx,
                    OpEx,
                    LCOF,
                    emissions,
                    comment,
                ]
                output_data.append(calculated_row)

        # Create a DataFrame for the output data
        output_df = pd.DataFrame(output_data, columns=output_columns)

        # Write the output data to a CSV file
        output_file = f"{process}_costs_emissions{suffix}.csv"
        output_df.to_csv(os.path.join(output_dir_process, output_file), index=False)

        print(
            f"Output CSV file created: {os.path.join(output_dir_process, output_file)}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cost/emissions and resource .csvs (optionally with component .jsons).")
    parser.add_argument(
        "-b", "--save_breakdowns", action="store_true",
        help="Save component-level CapEx, OpEx, and emissions breakdowns to JSON files"
    )
    parser.add_argument(
        "-y", "--year", type=int, default=None,
        help="Projection year for electricity and other dynamic inputs (only affects inputs stored in *_projection.csv if provided)"
    )
    parser.add_argument(
        "-d", "--include_demands", action="store_true",
        help="Also compute and write resource demand CSVs for each fuel"
    )
    args = parser.parse_args()
    main(save_breakdowns=args.save_breakdowns, year=args.year, include_demands=args.include_demands)
