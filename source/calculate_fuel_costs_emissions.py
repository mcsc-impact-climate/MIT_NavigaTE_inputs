"""
Date: Aug 21, 2024
Purpose: Prepare .csv files contained in input_fuel_pathway_data using consistent assumptions.
"""

import pandas as pd
import os

# ---------------------------------------------------------------------------
# One-time loaders wrapped in a helper – *nothing else calls these yet*
# ---------------------------------------------------------------------------
from common_tools import get_top_dir, ensure_directory_exists
from load_inputs import (
    load_global_parameters,
    load_molecular_info,
    load_technology_info,
)

# --------------------------------------------------------------------------------------
# 0.  LOAD ONE-TIME INPUTS  +  DERIVE REUSABLE NUMBERS (with full comments)
# --------------------------------------------------------------------------------------
from typing import Dict

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
    T = ctx["tech"]
    D = ctx["derived"]

    # ------------------------------------------------------------------
    # Natural-gas recovery and liquefaction from GREET 2024
    # ------------------------------------------------------------------
    ng_info = pd.read_csv(
        os.path.join(ctx["top_dir"], "input_fuel_pathway_data", "lng_inputs_GREET_processed.csv"),
        index_col="Stage",
    )

    D["NG_NG_demand_kg"] = ng_info.loc["Production", "NG Consumption (kg/kg)"]
    D["NG_NG_demand_GJ"] = ng_info.loc["Production", "NG Consumption (GJ/kg)"]
    D["NG_water_demand"] = ng_info.loc["Production", "Water Consumption (m^3/kg)"]
    D["NG_elec_demand"]  = ng_info.loc["Production", "Electricity Consumption (kWh/kg)"]
    D["NG_CO2_emissions"] = ng_info.loc["Production", "CO2 Emissions (kg/kg)"]
    D["NG_CH4_leakage"]   = ng_info.loc["Production", "CH4 Emissions (kg/kg)"]

    D["NG_liq_NG_demand_kg"] = ng_info.loc["Liquefaction", "NG Consumption (kg/kg)"]
    D["NG_liq_NG_demand_GJ"] = ng_info.loc["Liquefaction", "NG Consumption (GJ/kg)"]
    D["NG_liq_water_demand"] = ng_info.loc["Liquefaction", "Water Consumption (m^3/kg)"]
    D["NG_liq_elect_demand"]  = ng_info.loc["Liquefaction", "Electricity Consumption (kWh/kg)"]
    D["NG_liq_CO2_emissions"] = ng_info.loc["Liquefaction", "CO2 Emissions (kg/kg)"]
    D["NG_liq_CH4_leakage"]   = ng_info.loc["Liquefaction", "CH4 Emissions (kg/kg)"]

    # ------------------------------------------------------------------
    # DERIVED VALUES for hydrogen pathways
    # ------------------------------------------------------------------

    # ── STP H₂ via SMR (no capture) ────────────────────────────────────
    # Inputs from Zang et al 2024 (SMR case)
    smr_prod = T["H2_SMR"]["hourly_prod"]["value"]                      # [kg H₂/h]
    D["H2_SMR_elec_demand"] = T["H2_SMR"]["elec_cons"]["value"] / smr_prod  # [kWh elect/kg H₂]
    SMR_NG = T["H2_SMR"]["NG_cons"]["value"] - T["H2_SMR"]["steam_byproduct"]["value"] / T["H2_SMR"]["boiler_eff"]["value"]   # [GJ NG/hr]: NG consumption including steam displacement at 80% boiler efficiency
    D["H2_SMR_NG_demand"] = (                                              # [GJ NG/kg H₂]
        SMR_NG / smr_prod
        * (1 + D["NG_NG_demand_kg"])         # Also account for the additional NG consumed to process and recover the NG, from GREET 2024
    )
    D["H2_SMR_water_demand"] = T["H2_SMR"]["water_cons"]["value"] / smr_prod  # [m³ H₂O/kg H₂]
    # Base CapEx, converted from 2019 USD to 2024 USD then amortised (From Zang et al 2024 and H2A)
    D["H2_SMR_base_CapEx"] = (
        G["2019_to_2024_USD"]["value"] * T["H2_SMR"]["TPC_2019"]["value"]
        / 365 * T["H2_SMR"]["CRF"]["value"]
    )
    D["H2_SMR_onsite_emissions"] = T["H2_SMR"]["emissions"]["value"] / smr_prod   # [kg CO2e/kg H₂]
    D["H2_SMR_yearly_output"] = smr_prod * 24 * 365                               # [kg H₂/year]

    # ── STP H₂ via ATR-CC-S (99 % capture; ATR-CC-R-OC case) ───────────
    atr_prod = T["H2_ATRCCS"]["hourly_prod"]["value"]
    D["H2_ATRCCS_elec_demand"] = T["H2_ATRCCS"]["elec_cons"]["value"] / atr_prod  # [kWh elect/kg H₂]
    D["H2_ATRCCS_NG_demand"] = (                                                 # [GJ NG/kg H₂]
        T["H2_ATRCCS"]["NG_cons"]["value"] / atr_prod
        * (1 + D["NG_NG_demand_kg"])        # GREET 2024 uplift
    )
    D["H2_ATRCCS_water_demand"] = T["H2_ATRCCS"]["water_cons"]["value"] / atr_prod  # [m³ H₂O/kg H₂]
    # Amortised TPC – From Zang et al 2024 and H2A
    D["H2_ATRCCS_base_CapEx"] = (
        G["2019_to_2024_USD"]["value"] * T["H2_ATRCCS"]["TPC_2019"]["value"]
        / 365 * T["H2_ATRCCS"]["CRF"]["value"]
    )
    D["H2_ATRCCS_onsite_emissions"] = T["H2_ATRCCS"]["emissions"]["value"] / atr_prod  # [kg CO2e/kg H₂]
    D["H2_ATRCCS_yearly_output"] = atr_prod * 24 * 365                                  # [kg H₂/year]

    # ── STP H₂ via SMR-CCS (96 % capture) ─────────────────────────────
    smrccs_prod = T["H2_SMRCCS"]["hourly_prod"]["value"]
    D["H2_SMRCCS_elec_demand"] = T["H2_SMRCCS"]["elec_cons"]["value"] / smrccs_prod
    D["H2_SMRCCS_NG_demand"] = (
        T["H2_SMRCCS"]["NG_cons"]["value"] / smrccs_prod
        * (1 + D["NG_NG_demand_kg"])
    )
    D["H2_SMRCCS_water_demand"] = T["H2_SMRCCS"]["water_cons"]["value"] / smrccs_prod
    D["H2_SMRCCS_base_CapEx"] = (
        G["2019_to_2024_USD"]["value"] * T["H2_SMRCCS"]["TPC_2019"]["value"]
        / 365 * T["H2_SMRCCS"]["CRF"]["value"]
    )
    D["H2_SMRCCS_onsite_emissions"] = T["H2_SMRCCS"]["emissions"]["value"] / smrccs_prod
    D["H2_SMRCCS_yearly_output"] = smrccs_prod * 24 * 365

    # ── Low-Temperature Electrolysis (LTE) ─────────────────────────────
    D["H2_LTE_elec_demand"] = T["H2_LTE"]["elect_demand"]["value"]           # [kWh/kg H₂] from H2A
    D["H2_LTE_NG_demand"] = T["H2_LTE"]["NG_demand"]["value"]               # [GJ/kg H₂] aux boiler
    D["H2_LTE_water_demand"] = T["H2_LTE"]["water_demand"]["value"]         # [m³/kg H₂]
    D["H2_LTE_base_CapEx"] = T["H2_LTE"]["base_CapEx"]["value"]             # [2024 $ / kg H₂ y⁻¹]
    D["H2_LTE_onsite_emissions"] = T["H2_LTE"]["onsite_emissions"]["value"] # [kg CO2e/kg H₂]
    D["H2_LTE_yearly_output"] = T["H2_LTE"]["yearly_output"]["value"]       # [kg H₂/year]

    # ── Biomass Gasification (BG) – lignocellulosic ────────────────────
    D["H2_BG_elec_demand"] = T["H2_BG"]["elec_demand"]["value"]
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

CTX = build_context()          #  ← Used later; harmless for now
top_dir = CTX["top_dir"]       #  ← your old `top_dir` variable still works

# ---------------------------------------------------------------------------
# STEP-2:  static parameter table (tiny version – only H2_SMR for now)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Callable, Dict

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
        elect_demand=lambda c: c["derived"]["H2_SMR_elec_demand"],
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
        elect_demand=lambda c: c["derived"]["H2_ATRCCS_elec_demand"],
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
        elect_demand=lambda c: c["derived"]["H2_SMRCCS_elec_demand"],
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
        elect_demand=lambda c: c["derived"]["H2_LTE_elec_demand"],
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
        elect_demand=lambda c: c["derived"]["H2_BG_elec_demand"],
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
        elect_demand=lambda c: c["derived"]["NG_elec_demand"],
        lcb_demand=lambda c: 0.0,
        ng_demand=lambda c: c["glob"]["NG_HHV"]["value"] + c["derived"]["NG_NG_demand_GJ"],             # Additionally account for the NG consumed in producing the NG
        water_demand=lambda c: c["derived"]["NG_water_demand"],
        base_capex=lambda c: 0.0,                # CapEx = 0 for commodity NG
        employees=lambda c: 0,
        yearly_output=lambda c: 1.0,             # dummy (unused when employees=0)
        onsite_emiss=lambda c: 0.0,              # accounted upstream
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
        employees=lambda c: 0,            # assume incremental block has no labour
        yearly_output=lambda c: 1.0,
        onsite_emiss=lambda c: (
            c["derived"]["NG_liq_CO2_emissions"]
            + c["derived"]["NG_liq_CH4_leakage"] * c["glob"]["NG_GWP"]["value"]
        ),
    )
}

# --------------------------------------------------------------------------------------
# 2.  UNIT CONVERSIONS (centrelised)
# --------------------------------------------------------------------------------------
CONV = {
    "KG_PER_TONNE": 1_000,
    "MJ_PER_KWH": 3.6,
    "MJ_PER_GJ": 1_000,
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
    install_factor: float,
    prices: dict,
    elec_intensity: float,
    *,
    lcb_upstream_emiss: float = 0.0,   # keep args you’ll need later
) -> tuple[float, float, float]:
    """
    Pathway-agnostic production cost & emissions calculator.
    Returns (CapEx, OpEx, emissions) per *kg fuel*.

    Only uses the four basic demands stored in PATHWAYS:
    electricity, NG, water, LCB.
    """
    elect  = _p(path, "elect_demand")
    ng     = _p(path, "ng_demand")
    water  = _p(path, "water_demand")
    lcb    = _p(path, "lcb_demand")

    # -------- CapEx ----------------------------------------------------
    capex = _p(path, "base_capex") * install_factor

    # -------- OpEx -----------------------------------------------------
    fixed_opex = (
        CTX["glob"]["workhours_per_year"]["value"]
        * prices["labor"]
        * _p(path, "employees")
        / _p(path, "yearly_output")
        * (1 + CTX["glob"]["gen_admin_rate"]["value"])
        + (CTX["glob"]["op_maint_rate"]["value"] + CTX["glob"]["tax_rate"]["value"]) * capex
    )

    variable_opex = (
        elect  * prices["elec"]
        + ng   * prices["ng"]
        + water * prices["water"]
        + lcb   * prices["lcb"]
    )

    # -------- Emissions -------------------------------------------------
    emiss = (
        elect * elec_intensity
        # fugitive CH4 and upstream CO2-e
        + ng / CTX["glob"]["NG_HHV"]["value"] * CTX["glob"]["NG_GWP"]["value"] * CTX["derived"]["NG_CH4_leakage"]
        + ng / CTX["glob"]["NG_HHV"]["value"] * CTX["derived"]["NG_CO2_emissions"]
        + _p(path, "onsite_emiss")
        + lcb * lcb_upstream_emiss
    )

    return capex, fixed_opex + variable_opex, emiss
    
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
    elect  = _p(path, "elect_demand")  if include_elect else 0.0
    lcb    = _p(path, "lcb_demand")    * lcb_factor
    ng     = _p(path, "ng_demand")
    water  = _p(path, "water_demand")
    # all STP fuels in this repo consume no external CO₂
    return elect, lcb, ng, water, 0.0

# ─────────────────────────────────────────────────────────────
# Incremental H₂ liquefaction  (adds only extra electricity)
# ─────────────────────────────────────────────────────────────
def demands_liquid_h2(H_pathway: str):
    elect, lcb, ng, water, co2 = generic_demands(H_pathway)
    elect += tech_info["H2_liquefaction"]["elec_demand"]["value"]
    return elect, lcb, ng, water, co2


# ─────────────────────────────────────────────────────────────
# Incremental H₂ compression  (adds only extra electricity)
# ─────────────────────────────────────────────────────────────
def demands_compressed_h2(H_pathway: str):
    elect, lcb, ng, water, co2 = generic_demands(H_pathway)
    elect += tech_info["H2_compression"]["elec_demand"]["value"]
    return elect, lcb, ng, water, co2


# ---------------------------------------------------------------------------
# Feed-stock helpers: generic H₂ and CO₂ (BEC / DAC / fossil) in one place
# ---------------------------------------------------------------------------
def _feed_h2(
    h_path: str,
    instal: float,
    prices: dict,
    elec_int: float,
    lcb_up_emiss: float,
) -> tuple[float, float, float]:
    """CapEx, OpEx, Emiss for 1 kg of *STP hydrogen* from pathway *h_path*."""
    return generic_production(
        h_path, instal, prices, elec_int, lcb_upstream_emiss=lcb_up_emiss
    )


# ----------------------------------------------------------------------
# Helper: cost / emissions for 1 kg of captured-CO₂ feedstock
# ----------------------------------------------------------------------
def _feed_co2(
    C_pathway: str,
    credit_per_kg_fuel: float,   # e.g. MW_CO2 / MW_MeOH   or  nC*MW_CO2 / MW_FTdiesel
    prices: dict,
    elect_int: float,
    CO2_demand: float,
):
    """Return (capex, opex, emissions) *per kg CO₂* for any fuel.

    `credit_per_kg_fuel` is the (negative) credit to apply for 1 kg of fuel
    when carbon stays sequestered in the product rather than emitted.
    """
    if C_pathway == "BEC":
        cap = 0.0
        op  = BEC_CO2_price
        em  = BEC_CO2_upstream_emissions * CO2_demand - credit_per_kg_fuel
    elif C_pathway == "DAC":
        cap = 0.0
        op  = DAC_CO2_price
        em  = (
            DAC_CO2_upstream_emissions * CO2_demand
            + DAC_CO2_upstream_NG * CO2_demand / NG_HHV * NG_GWP * NG_CH4_leakage
            + DAC_CO2_upstream_elect * elect_int * CO2_demand
            - credit_per_kg_fuel
        )
    elif C_pathway in ("SMRCCS", "ATRCCS"):
        cap = 0.0
        op  = 0.0
        em  = CO2_demand - credit_per_kg_fuel      # captured fossil CO₂, not 100 % conv.
    elif C_pathway == "SMR":
        cap = 0.0
        op  = 0.0
        em  = - credit_per_kg_fuel                 # fossil CO₂ retained in fuel (credit)
    elif C_pathway == "BG":
        cap = 0.0
        op  = 0.0
        em  = 0.0                                  # biogenic credit already counted
    else:
        raise ValueError(f"Unknown CO₂ pathway: {C_pathway}")

    return cap, op, em



def calculate_BEC_upstream_emission_rate(filename = f"{top_dir}/input_fuel_pathway_data/BEC_upstream_emissions_GREET.csv"):
    """
    Calculates the upstream emissions for CO2 captured from a bioenergy plant (in kg CO2e / kg CO2), by averaging over GREET estimates for different US states
    """

    # Read in the BEC emissions data from GREET
    BEC_emissions_data = pd.read_csv(filename)

    # Calculate the upstream emission rate for each state
    BEC_emissions_data["Upstream emissions (kg CO2e / kg CO2"] = (BEC_emissions_data["Feedstock emissions (g CO2e/mmBtu)"] + BEC_emissions_data["Fuel emissions (g CO2e/mmBtu)"]) / BEC_emissions_data["CO2 from CCS"]

    # Calculate the average upstream emissions rate over all states
    average_upstream_emissions_rate = BEC_emissions_data["Upstream emissions (kg CO2e / kg CO2"].mean()

    return average_upstream_emissions_rate

def calculate_DAC_upstream_resources_emissions(material_reqs_filename=f"{top_dir}/input_fuel_pathway_data/DAC_material_reqs.csv", upstream_elec_NG_filename=f"{top_dir}/input_fuel_pathway_data/DAC_upstream_electricity_NG.csv"):

    ############### Calculate emissions and resource demands associated with CO2 capture and compression ###############
    upstream_elec_NG_info = pd.read_csv(upstream_elec_NG_filename)

    # Collect upstream electricity demand (capture + compression)
    upstream_elec = upstream_elec_NG_info["Electricity for CO2 capture (MJ/MT-CO2)"][0] + upstream_elec_NG_info["Electricity for CO2 compression at the CO2 source (MJ/MT-CO2)"][0]

    # Convert from MJ / MT CO2 to kWh / kg CO2
    KG_PER_TONNE = 1000
    KG_PER_TON = 907.185
    MJ_PER_KWH = 3.6
    upstream_elec = upstream_elec / (MJ_PER_KWH * KG_PER_TONNE)

    # Collect upstream NG consumption associated with CO2 capture
    upstream_NG = upstream_elec_NG_info["Natural gas for CO2 capture (MJ/MT-CO2)"][0]

    # Convert from MJ / MT CO2 to GJ / kg CO2
    MJ_PER_GJ = 1000
    upstream_NG = upstream_NG / (MJ_PER_GJ * KG_PER_TONNE)
    ####################################################################################################################

    ################## Calculate emissions and resource demands embedded in the carbon capture plant ###################
    material_reqs_info = pd.read_csv(material_reqs_filename)

    # Convert water consumption from gals / ton to m^3 / kg CO2 for each material
    GAL_PER_CBM = 264.172
    material_reqs_info["Water consumption (m^3/kg-CO2)"] = (material_reqs_info["Water consumption (gals/ton)"] / (GAL_PER_CBM * KG_PER_TON)) * (material_reqs_info["kg / ton CO2"] / KG_PER_TON)

    # Convert NG consumption from mmBtu/ton to GJ / kg CO2 for each material
    BTU_PER_MJ = 947.817
    BTU_PER_MMBTU = 1e6
    material_reqs_info["NG consumption (GJ/kg-CO2)"] = (material_reqs_info["NG consumption (mmBtu/ton)"] * (BTU_PER_MMBTU) / (BTU_PER_MJ * MJ_PER_GJ * KG_PER_TON)) * (material_reqs_info["kg / ton CO2"] / KG_PER_TON)

    # Convert GHG emissions from g CO2e/ton to kg CO2e/kg CO2 for each material
    G_PER_KG = 1000
    material_reqs_info["GHG emissions (kg CO2e/kg-CO2)"] = (material_reqs_info["GHG emissions (g CO2e/ton)"] / (G_PER_KG * KG_PER_TON)) * (material_reqs_info["kg / ton CO2"] / KG_PER_TON)

    # Sum over all materials to get embedded water, NG, and GHG emissions
    embedded_water = material_reqs_info["Water consumption (m^3/kg-CO2)"].sum()
    embedded_NG = material_reqs_info["NG consumption (GJ/kg-CO2)"].sum()
    embedded_emissions = material_reqs_info["GHG emissions (kg CO2e/kg-CO2)"].sum()

    upstream_water = embedded_water
    upstream_NG = upstream_NG + embedded_NG
    upstream_emissions = embedded_emissions

    return upstream_emissions, upstream_elec, upstream_NG, upstream_water
    ####################################################################################################################

calculate_DAC_upstream_resources_emissions()

########################################## Read in NG production inputs ############################################
NG_info = pd.read_csv(f"{top_dir}/input_fuel_pathway_data/lng_inputs_GREET_processed.csv", index_col="Stage")
NG_water_demand = NG_info.loc["Production", "Water Consumption (m^3/kg)"] # [m^3 H2O / kg NG]. Source: GREET 2024
NG_NG_demand_GJ = NG_info.loc["Production", "NG Consumption (GJ/kg)"] # [GJ NG consumed / kg NG produced]. Source: GREET 2024
NG_elect_demand = NG_info.loc["Production", "Electricity Consumption (kWh/kg)"] # [GJ NG consumed / kg NG produced]. Source: GREET 2024
NG_NG_demand_kg = NG_info.loc["Production", "NG Consumption (kg/kg)"] # [kg NG consumed / kg NG produced]. Source: GREET 2024
NG_CO2_emissions = NG_info.loc["Production", "CO2 Emissions (kg/kg)"] # [kg CO2 / kg NG]. Source: GREET 2024
NG_CH4_leakage = NG_info.loc["Production", "CH4 Emissions (kg/kg)"] # [kg CH4 / kg NG]. Source: GREET 2024
####################################################################################################################

############################################# Read in global parameters ############################################
global_parameters = load_global_parameters()
workhours_per_year = global_parameters["workhours_per_year"]["value"]
gen_admin_rate = global_parameters["gen_admin_rate"]["value"]
op_maint_rate = global_parameters["op_maint_rate"]["value"]
tax_rate = global_parameters["tax_rate"]["value"]

NG_HHV = global_parameters["NG_HHV"]["value"]
NG_GWP = global_parameters["NG_GWP"]["value"]

BEC_CO2_price = global_parameters["BEC_CO2_price"]["value"]
BEC_CO2_upstream_emissions = calculate_BEC_upstream_emission_rate() # [kg CO2e/kg CO2] upstream emissions from bioenergy plant with CO2 capture (e.g. 0.02 from biomass BEC, 0.05 from biogas BEC..)

DAC_CO2_price = global_parameters["DAC_CO2_price"]["value"]
DAC_upstream_emissions, DAC_upstream_elect, DAC_upstream_NG, DAC_upstream_water = calculate_DAC_upstream_resources_emissions()
DAC_CO2_upstream_emissions = DAC_upstream_emissions # [kg CO2e/kg CO2] upstream emissions from direct-air CO2 capture, from GREET 2024, accounting for both operational and embedded emissions
DAC_CO2_upstream_NG = DAC_upstream_NG   # From GREET 2024, accounting for both operational and embedded emissions
DAC_CO2_upstream_water = DAC_upstream_water     # From GREET 2024, accounting for both operational and embedded emissions
DAC_CO2_upstream_elect = DAC_upstream_elect     # From GREET 2024, accounting for operational electricity consumption
####################################################################################################################

############################################## Read in molecular info ##############################################
molecular_info = load_molecular_info()
MW_CO2 = molecular_info["MW_CO2"]["value"]
MW_MeOH = molecular_info["MW_MeOH"]["value"]
MW_H2 = molecular_info["MW_H2"]["value"]
MW_NH3 = molecular_info["MW_NH3"]["value"]
MW_FTdiesel = molecular_info["MW_FTdiesel"]["value"]
nC_FTdiesel = molecular_info["nC_FTdiesel"]["value"]
####################################################################################################################

############################## Read in technology info and calculate derived parameters ############################
tech_info = load_technology_info()

####################################### Inputs for NG liquefaction ########################################
NG_liq_base_CapEx = 0.82 # [2024$/kg NG]. Obtained from Table 3 (USA Lower 48) in https://www.jstor.org/stable/resrep31040.11?seq=7 and converted from 2018$ to 2024$ using https://data.bls.gov/cgi-bin/cpicalc.pl?cost1=100&year1=201901&year2=202401
NG_liq_NG_demand_GJ = NG_info.loc["Liquefaction", "NG Consumption (GJ/kg)"] # [GJ NG consumed / kg liquefied NG]. Source: GREET 2024
NG_liq_NG_demand_kg = NG_info.loc["Liquefaction", "NG Consumption (kg/kg)"] # [kg NG consumed / kg liquefied NG]. Source: GREET 2024
NG_liq_water_demand = NG_info.loc["Liquefaction", "Water Consumption (m^3/kg)"] # [m^3 H2O / kg NG]. Source: GREET 2024
NG_liq_elect_demand = NG_info.loc["Liquefaction", "Electricity Consumption (kWh/kg)"] # [m^3 H2O / kg NG]. Source: GREET 2024
NG_liq_CO2_emissions = NG_info.loc["Liquefaction", "CO2 Emissions (kg/kg)"] # [kg CO2 / kg NG]. Source: GREET 2024
NG_liq_CH4_leakage = NG_info.loc["Liquefaction", "CH4 Emissions (kg/kg)"] # [kg CH4 / kg NG]. Source: GREET 2024
###########################################################################################################

################### Inputs for liquid NH3 production from arbitrary H2 feedstock ##########################
NH3_H2_demand = 3/2 * MW_H2 / MW_NH3 # [kg H2/kg NH3] stoichiometry
NH3_elect_demand = tech_info["NH3"]["elect_demand"]["value"] - tech_info["H2_LTE"]["elect_demand"]["value"]*NH3_H2_demand # subtract electrical demand from LTE H2 process
NH3_water_demand = tech_info["NH3"]["water_demand_LTE"]["value"] - tech_info["H2_LTE"]["water_demand"]["value"]*NH3_H2_demand # subtract water demand from LTE H2 process
NH3_base_CapEx = tech_info["NH3"]["base_CapEx_LTE"]["value"] - tech_info["H2_LTE"]["base_CapEx"]["value"]*NH3_H2_demand # subtract base CapEx from LTE H2 process
NH3_full_time_employed = 10 # [full-time employees] from H2A
NH3_full_time_employed = tech_info["NH3"]["employees_LTE"]["value"] - tech_info["H2_LTE"]["employees"]["value"] # subtract employees from LTE H2 process
###########################################################################################################
    
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
    capex, opex, emiss = generic_production(
        H_pathway,
        instal_factor,
        prices,
        elect_emissions_intensity,
        lcb_upstream_emiss=LCB_upstream_emissions,
    )
    return capex, opex, emiss
    
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
    base_capex_liq   = tech_info["H2_liquefaction"]["base_CapEx"]["value"]
    elect_liq        = tech_info["H2_liquefaction"]["elec_demand"]["value"]

    capex_liq  = base_capex_liq * instal_factor
    opex_liq   = (op_maint_rate + tax_rate) * capex_liq + elect_liq * elect_price
    emiss_liq  = elect_liq * elect_emissions_intensity

    # --- underlying STP hydrogen via generic engine --------------------
    prices = {
        "water": water_price,
        "ng":    NG_price,
        "lcb":   LCB_price,
        "elec":  elect_price,
        "labor": hourly_labor_rate,
    }
    capex_h2, opex_h2, emiss_h2 = generic_production(
        H_pathway,
        instal_factor,
        prices,
        elect_emissions_intensity,
        lcb_upstream_emiss=LCB_upstream_emissions,
    )

    return capex_liq + capex_h2, opex_liq + opex_h2, emiss_liq + emiss_h2

def calculate_production_costs_emissions_NG(
    water_price, NG_price, elect_price, elect_intensity
):
    prices = {
        "water": water_price,
        "ng":    NG_price,
        "lcb":   0.0,
        "elec":  elect_price,
        "labor": 0.0,
    }
    return generic_production(
        "NG",          # canonical table key
        install_factor=1.0,
        prices=prices,
        elec_intensity=elect_intensity,
    )
    
def calculate_production_costs_emissions_liquid_NG(
    instal_factor,
    water_price,
    NG_price,
    elect_price,
    elect_intensity,
):
    """Incremental liquefaction block + generic NG feedstock."""
    # incremental liquefaction
    capex_liq  = tech_info["NG_liq"]["base_CapEx_2018"]["value"] * global_parameters["2018_to_2024_USD"]["value"] * instal_factor
    opex_liq   = (
        (op_maint_rate + tax_rate) * capex_liq
        + CTX["derived"]["NG_liq_NG_demand_GJ"] * NG_price
        + CTX["derived"]["NG_liq_water_demand"] * water_price
        + CTX["derived"]["NG_liq_elect_demand"] * elect_price
    )
    emiss_liq  = (
        CTX["derived"]["NG_liq_CO2_emissions"]
        + CTX["derived"]["NG_liq_CH4_leakage"] * CTX["glob"]["NG_GWP"]["value"]
        + CTX["derived"]["NG_liq_elect_demand"] * elect_intensity
    )

    # upstream NG via generic engine
    capex_ng, opex_ng, emiss_ng = calculate_production_costs_emissions_NG(
        water_price, NG_price, elect_price, elect_intensity
    )

    return capex_liq + capex_ng, opex_liq + opex_ng, emiss_liq + emiss_ng

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
    base_capex_comp = tech_info["H2_compression"]["base_CapEx"]["value"]
    elect_comp      = tech_info["H2_compression"]["elec_demand"]["value"]

    capex_comp = base_capex_comp * instal_factor
    opex_comp  = (op_maint_rate + tax_rate) * capex_comp + elect_comp * elect_price
    emiss_comp = elect_comp * elect_emissions_intensity

    # --- underlying STP hydrogen via generic engine --------------------
    prices = {
        "water": water_price,
        "ng":    NG_price,
        "lcb":   LCB_price,
        "elec":  elect_price,
        "labor": hourly_labor_rate,
    }
    capex_h2, opex_h2, emiss_h2 = generic_production(
        H_pathway,
        instal_factor,
        prices,
        elect_emissions_intensity,
        lcb_upstream_emiss=LCB_upstream_emissions,
    )

    return capex_comp + capex_h2, opex_comp + opex_h2, emiss_comp + emiss_h2

# ────────────────────────────────────────────────────────────────────
# Liquid NH₃ (cryogenic, 1 bar) – flexible H₂ feed
# ────────────────────────────────────────────────────────────────────
def calculate_production_costs_emissions_ammonia(
    H_pathway: str,
    instal: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_up_emiss: float,
    elect_price: float,
    elect_int: float,
    labor_rate: float,
):
    # --- core Haber-Bosch block (no feeds) ----------------------------
    elect  = NH3_elect_demand                     # kWh / kg NH₃
    water  = NH3_water_demand                     # m³ / kg NH₃
    ng     = tech_info["NH3"]["NG_demand"]["value"]   # GJ / kg NH₃

    employees     = NH3_full_time_employed
    yearly_output = tech_info["NH3"]["yearly_output"]["value"]

    cap_core = NH3_base_CapEx * instal
    op_core  = (
        _fixed_opex(cap_core, employees, yearly_output, labor_rate)
        + elect * elect_price
        + water * water_price
        + ng    * NG_price
    )
    em_core  = elect * elect_int + ng / NG_HHV * NG_GWP * NG_CH4_leakage
    # (No onsite-process CO₂ for NH₃, so nothing else to add)

    # --- H₂ feed (generic engine) ------------------------------------
    H2_demand = NH3_H2_demand                     # kg H₂ per kg NH₃

    price_ctx = {
        "water": water_price,
        "ng":    NG_price,
        "lcb":   LCB_price,
        "elec":  elect_price,
        "labor": labor_rate,
    }

    cap_h2, op_h2, em_h2 = _feed_h2(
        H_pathway, instal, price_ctx, elect_int, LCB_up_emiss
    )

    # --- totals per kg NH₃ -------------------------------------------
    CapEx = cap_core + cap_h2 * H2_demand
    OpEx  = op_core  + op_h2  * H2_demand
    Emiss = em_core  + em_h2  * H2_demand

    return CapEx, OpEx, Emiss



def _fixed_opex(capex: float, employees: int, yearly_output: float,
                labor_rate: float) -> float:
    """Legacy fixed-OpEx block (labour + gen-admin + O&M + tax)."""
    labour = (
        workhours_per_year * labor_rate * employees / yearly_output
        * (1 + gen_admin_rate)
    )
    return labour + (op_maint_rate + tax_rate) * capex


def calculate_production_costs_emissions_methanol(
    H_pathway: str,
    C_pathway: str,
    instal: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_up_emiss: float,
    elect_price: float,
    elect_int: float,
    labor_rate: float,
):
    # ── “core” MeOH synthesis block (no feeds) ────────────────────────
    elect  = tech_info["MeOH"]["elect_demand"]["value"]
    water  = tech_info["MeOH"]["water_demand"]["value"]
    ng     = tech_info["MeOH"]["NG_demand"]["value"]
    employees     = tech_info["MeOH"]["employees"]["value"]
    yearly_output = tech_info["MeOH"]["yearly_output"]["value"]

    cap_core = tech_info["MeOH"]["base_CapEx"]["value"] * instal
    op_core  = (
        _fixed_opex(cap_core, employees, yearly_output, labor_rate)
        + elect * elect_price
        + water * water_price
        + ng    * NG_price
    )
    em_core  = elect * elect_int + ng / NG_HHV * NG_GWP * NG_CH4_leakage
    if C_pathway not in ("BEC", "DAC"):          # keep legacy rule
        em_core += tech_info["MeOH"]["onsite_emissions"]["value"]

    # ── H₂ and CO₂ feed handling via helpers ──────────────────────────
    H2_demand  = tech_info["MeOH"]["H2_demand"]["value"]
    CO2_demand = tech_info["MeOH"]["CO2_demand"]["value"]

    price_ctx = {
        "water": water_price,
        "ng":    NG_price,
        "lcb":   LCB_price,
        "elec":  elect_price,
        "labor": labor_rate,
    }

    cap_h2, op_h2, em_h2   = _feed_h2(
        H_pathway, instal, price_ctx, elect_int, LCB_up_emiss
    )
    
    credit_methanol = MW_CO2 / MW_MeOH
    cap_co2, op_co2, em_co2 = _feed_co2(
        C_pathway, credit_methanol, price_ctx, elect_int, CO2_demand
    )

    # ── totals (per kg MeOH) ──────────────────────────────────────────
    CapEx = cap_core + cap_h2 * H2_demand + cap_co2 * CO2_demand
    OpEx  = op_core  + op_h2  * H2_demand + op_co2  * CO2_demand
    Emiss = em_core  + em_h2  * H2_demand + em_co2

    return CapEx, OpEx, Emiss


# -----------------------------------------------------------------------
# Fischer–Tropsch diesel  (CxHyOz)  –  flexible H₂ & CO₂ feeds
# -----------------------------------------------------------------------
def calculate_production_costs_emissions_FTdiesel(
    H_pathway: str,
    C_pathway: str,
    instal: float,
    water_price: float,
    NG_price: float,
    LCB_price: float,
    LCB_up_emiss: float,
    elect_price: float,
    elect_int: float,
    labor_rate: float,
):
    # ── “core” FT block (no feeds) ────────────────────────────────────
    elect  = tech_info["FTdiesel"]["elect_demand"]["value"]
    water  = tech_info["FTdiesel"]["water_demand"]["value"]
    ng     = tech_info["FTdiesel"]["NG_demand"]["value"]
    employees     = tech_info["FTdiesel"]["employees"]["value"]
    yearly_output = tech_info["FTdiesel"]["yearly_output"]["value"]

    cap_core = tech_info["FTdiesel"]["base_CapEx"]["value"] * instal
    op_core  = (
        _fixed_opex(cap_core, employees, yearly_output, labor_rate)
        + elect * elect_price
        + water * water_price
        + ng    * NG_price
    )
    em_core  = elect * elect_int + ng / NG_HHV * NG_GWP * NG_CH4_leakage
    if C_pathway not in ("BEC", "DAC"):
        em_core += tech_info["FTdiesel"]["onsite_emissions"]["value"]

    # ── H₂ and CO₂ feeds ──────────────────────────────────────────────
    H2_demand  = tech_info["FTdiesel"]["H2_demand"]["value"]
    CO2_demand = tech_info["FTdiesel"]["CO2_demand"]["value"]

    price_ctx = {
        "water": water_price,
        "ng":    NG_price,
        "lcb":   LCB_price,
        "elec":  elect_price,
        "labor": labor_rate,
    }

    cap_h2, op_h2, em_h2 = _feed_h2(
        H_pathway, instal, price_ctx, elect_int, LCB_up_emiss
    )

    # fuel-specific credit factor:  nC * MW_CO2  /  MW_FTdiesel
    credit_ft = nC_FTdiesel * MW_CO2 / MW_FTdiesel
    cap_co2, op_co2, em_co2 = _feed_co2(
        C_pathway, credit_ft, price_ctx, elect_int, CO2_demand
    )

    # ── totals (per kg FT-diesel) ─────────────────────────────────────
    CapEx = cap_core + cap_h2 * H2_demand + cap_co2 * CO2_demand
    OpEx  = op_core  + op_h2  * H2_demand + op_co2  * CO2_demand
    Emiss = em_core  + em_h2  * H2_demand + em_co2

    return CapEx, OpEx, Emiss


def calculate_resource_demands_STP_hydrogen(H_pathway: str):
    """Return electricity, LCB, NG, water, CO₂ per kg H₂ for *H_pathway*."""
    elect = _p(H_pathway, "elect_demand")
    lcb = _p(H_pathway, "lcb_demand")
    ng = _p(H_pathway, "ng_demand")
    water = _p(H_pathway, "water_demand")
    return elect, lcb, ng, water, 0.0  # STP hydrogen never consumes external CO₂

def calculate_resource_demands_liquid_hydrogen(H_pathway):
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += tech_info["H2_liquefaction"]["elec_demand"]["value"]

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_compressed_hydrogen(H_pathway):
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += tech_info["H2_compression"]["elec_demand"]["value"]

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand

def calculate_resource_demands_NG():
    water_demand = NG_water_demand      # m^3 water / kg NG
    NG_demand = NG_HHV + NG_NG_demand_GJ     # GJ NG / kg NG
    elect_demand = NG_elect_demand
    LCB_demand = 0
    CO2_demand = 0

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand

def calculate_resource_demands_liquid_NG():
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_NG()
    NG_demand += NG_liq_NG_demand_GJ
    water_demand += NG_liq_water_demand
    elect_demand += NG_liq_elect_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand

def calculate_resource_demands_ammonia(H_pathway):
    elect_demand = NH3_elect_demand
    LCB_demand = 0
    H2_demand = NH3_H2_demand
    CO2_demand = 0
    NG_demand = tech_info["NH3"]["NG_demand"]["value"]
    water_demand = NH3_water_demand

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += H2_elect_demand * H2_demand
    LCB_demand += H2_LCB_demand * H2_demand
    NG_demand += H2_NG_demand * H2_demand
    water_demand += H2_water_demand * H2_demand
    CO2_demand += H2_CO2_demand * H2_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_methanol(H_pathway, C_pathway):
    elect_demand = tech_info["MeOH"]["elect_demand"]["value"]
    LCB_demand = tech_info["MeOH"]["LCB_demand"]["value"]
    H2_demand = tech_info["MeOH"]["H2_demand"]["value"]
    CO2_demand = tech_info["MeOH"]["CO2_demand"]["value"]
    NG_demand = tech_info["MeOH"]["NG_demand"]["value"]
    water_demand = tech_info["MeOH"]["water_demand"]["value"]

    if C_pathway=="DAC":
        NG_demand = NG_demand + DAC_CO2_upstream_NG * CO2_demand
        water_demand = water_demand + DAC_CO2_upstream_water * CO2_demand
        elect_demand = elect_demand + DAC_CO2_upstream_elect * CO2_demand

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += H2_elect_demand * H2_demand
    LCB_demand += H2_LCB_demand * H2_demand
    NG_demand += H2_NG_demand * H2_demand
    water_demand += H2_water_demand * H2_demand
    CO2_demand += H2_CO2_demand * H2_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand

def calculate_resource_demands_FTdiesel(H_pathway, C_pathway):
    elect_demand = tech_info["FTdiesel"]["elect_demand"]["value"]
    LCB_demand = tech_info["FTdiesel"]["LCB_demand"]["value"]
    H2_demand = tech_info["FTdiesel"]["H2_demand"]["value"]
    CO2_demand = tech_info["FTdiesel"]["CO2_demand"]["value"]
    NG_demand = tech_info["FTdiesel"]["NG_demand"]["value"]
    water_demand = tech_info["FTdiesel"]["water_demand"]["value"]

    if C_pathway=="DAC":
        NG_demand = NG_demand + DAC_CO2_upstream_NG * CO2_demand
        water_demand = water_demand + DAC_CO2_upstream_water * CO2_demand
        elect_demand = elect_demand + DAC_CO2_upstream_elect * CO2_demand

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
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
}

fuel_comments = {
    "hydrogen": "hydrogen at standard temperature and pressure",
    "liquid_hydrogen": "Liquid cryogenic hydrogen at atmospheric pressure",
    "compressed_hydrogen": "compressed gaseous hydrogen at 700 bar",
    "ammonia": "Liquid cryogenic ammonia at atmospheric pressure",
    "methanol": "liquid methanol at STP",
    "FTdiesel": "liquid Fischer--Tropsch diesel fuel at STP",
    "ng": "natural gas at standard temperature and pressure",
    "lng": "liquid natural gas at atmospheric pressure",
}


def main():
    input_dir = f"{top_dir}/input_fuel_pathway_data/"
    output_dir_production = f"{top_dir}/input_fuel_pathway_data/production/"
    ensure_directory_exists(output_dir_production)
    output_dir_process = f"{top_dir}/input_fuel_pathway_data/process/"
    ensure_directory_exists(output_dir_process)

    # Read the input CSV files
    input_df = pd.read_csv(input_dir + 'regional_TEA_inputs.csv')
    pathway_df = pd.read_csv(input_dir + 'fuel_pathway_options.csv')

    # Populate the arrays using the columns
    fuels_and_contents = pathway_df[['WTG fuels', 'fuel contents']].dropna()
    fuels = fuels_and_contents['WTG fuels'].tolist()
    fuel_content_map = dict(zip(fuels_and_contents['WTG fuels'], fuels_and_contents['fuel contents']))
    processes = pathway_df['GTT processes'].dropna().tolist()
    
    def get_unique_sources(column):
        return sorted(set(
            source.strip()
            for sources in pathway_df[column].dropna()
            for source in str(sources).split(",")
            if source.strip()
        ))
    
    Esources = get_unique_sources('electricity sources')
    Hsources = get_unique_sources('hydrogen sources')
    Csources = get_unique_sources('carbon sources')
    # Well to Gate fuel production
    for fuel in fuels:
        if "fossil" in fuel_content_map[fuel]:
            fuel_pathways=["fossil"]
            H_pathways = ["n/a"]
            C_pathways = ["n/a"]
            E_pathways = ["n/a"]
        else:
            if "C" in fuel_content_map[fuel]: # if fuel contains carbon
                fuel_pathways_noelec = []
                H_pathways_noelec = []
                C_pathways_noelec = []
                for Csource in Csources:
                    for Hsource in Hsources:
                        if Hsource == Csource:
                            H_pathways_noelec += [Hsource]
                            C_pathways_noelec += [Csource]
                            fuel_pathways_noelec += [Hsource + "_H_C"]
                        elif ((Hsource == "SMR") & ((Csource == "SMRCCS") | (Csource == "ATRCCS"))) | ((Hsource != "SMR") & (Csource == "SMR")) | ((Hsource != "BG") & (Csource == "BG")):
                            #skip case where ATRCCS/SMRCCS is used for C and SMR is used for H, because this does not make sense.
                            #also skip cases where BG or SMR is used for C but not H, because C would not be captured or usable in those cases.
                            continue
                        else:
                            H_pathways_noelec += [Hsource]
                            C_pathways_noelec += [Csource]
                            fuel_pathways_noelec += [Hsource + "_H_" + Csource + "_C"]
            else: # fuel does not contain carbon
                fuel_pathways_noelec = [Hsource + "_H" for Hsource in Hsources]
                H_pathways_noelec = [Hsource for Hsource in Hsources]
                C_pathways_noelec = ["n/a" for Hsource in Hsources]
            fuel_pathways = []
            H_pathways = []
            C_pathways = []
            E_pathways = []
            for Esource in Esources:
                H_pathways += [H_pathway_noelec for H_pathway_noelec in H_pathways_noelec]
                C_pathways += [C_pathway_noelec for C_pathway_noelec in C_pathways_noelec]
                E_pathways += [Esource for fuel_pathway_noelec in fuel_pathways_noelec]
                fuel_pathways += [fuel_pathway_noelec + "_" + Esource + "_E" for fuel_pathway_noelec in fuel_pathways_noelec]

        # List to hold all rows for the output CSV
        output_data = []
        # GE - list to hold all resource rows for the ouput csv
        output_resource_data = []

        # Iterate through each row in the input data and perform calculations
        for fuel_pathway in fuel_pathways:
            pathway_index = fuel_pathways.index(fuel_pathway)
            H_pathway = H_pathways[pathway_index]
            C_pathway = C_pathways[pathway_index]

            if fuel == "ng":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = generic_demands("NG")
                comment = "natural gas at standard temperature and pressure"
            elif fuel == "lng":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = generic_demands("NG")
                elect_demand += CTX["derived"]["NG_liq_elect_demand"]
                NG_demand    += CTX["derived"]["NG_liq_NG_demand_GJ"]
                water_demand += CTX["derived"]["NG_liq_water_demand"]
                comment = "liquid natural gas at atmospheric pressure"

            # Get resource demand function and call it
            if fuel in resource_demand_fn:
                if fuel in ["methanol", "FTdiesel"]:
                    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = resource_demand_fn[fuel](H_pathway, C_pathway)
                elif fuel in ["ammonia"]:
                    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = resource_demand_fn[fuel](H_pathway)
                elif fuel in ["ng", "lng"]:
                    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = resource_demand_fn[fuel]()
                else:
                    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = resource_demand_fn[fuel](H_pathway)

            calculated_resource_row = [fuel, H_pathway, C_pathway, fuel_pathway, elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand]
            output_resource_data.append(calculated_resource_row)

            for row_index, row in input_df.iterrows():
                region,instal_factor_low,instal_factor_high,src,water_price,src,NG_price,src,NG_fugitive_emissions,src,LCB_price,src,LCB_upstream_emissions,src,grid_price,src,grid_emissions_intensity,src,solar_price,src,solar_emissions_intensity,src,wind_price,src,wind_emissions_intensity,src,nuke_price,src,nuke_emissions_intensity,src,hourly_labor_rate,src = row

                # Calculate the average installation factor
                instal_factor = (instal_factor_low + instal_factor_high) / 2
                H_pathway = H_pathways[pathway_index]
                C_pathway = C_pathways[pathway_index]
                E_pathway = E_pathways[pathway_index]
                # Check whether we are working with grid or renewable electricity
                # if renewables, fix electricity price and emissions to imposed values
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
                    
                # Get the appropriate function for cost/emissions
                if fuel in cost_emission_fn:
                    if fuel in ["methanol", "FTdiesel"]:
                        CapEx, OpEx, emissions = cost_emission_fn[fuel](
                            H_pathway, C_pathway, instal_factor,
                            water_price, NG_price, LCB_price, LCB_upstream_emissions,
                            elect_price, elect_emissions_intensity, hourly_labor_rate
                        )
                    elif fuel in ["ammonia"]:
                        CapEx, OpEx, emissions = cost_emission_fn[fuel](
                            H_pathway, instal_factor,
                            water_price, NG_price, LCB_price, LCB_upstream_emissions,
                            elect_price, elect_emissions_intensity, hourly_labor_rate
                        )
                    elif fuel == "ng":
                        CapEx, OpEx, emissions = cost_emission_fn[fuel](
                            water_price, NG_price, elect_price, elect_emissions_intensity
                        )
                    elif fuel == "lng":
                        CapEx, OpEx, emissions = cost_emission_fn[fuel](
                            instal_factor, water_price, NG_price, elect_price, elect_emissions_intensity
                        )
                    else:
                        CapEx, OpEx, emissions = cost_emission_fn[fuel](
                            H_pathway, instal_factor,
                            water_price, NG_price, LCB_price, LCB_upstream_emissions,
                            elect_price, elect_emissions_intensity, hourly_labor_rate
                        )
                    # Retrieve the descriptive comment string
                    comment = fuel_comments.get(fuel, "")

                CapEx *= 1000 # convert to $/tonne
                OpEx *= 1000 # convert to $/tonne
                LCOF = CapEx + OpEx # in $/tonne
                calculated_row = [fuel, H_pathway, C_pathway, E_pathway, fuel_pathway, region, 1, 2024, CapEx, OpEx, LCOF, emissions, comment]
                output_data.append(calculated_row)


        # GE - Define the resource output to CSV column names - may need to add more columns
        output_resource_columns = [
            "Fuel", "Hydrogen Source", "Carbon Source", "Fuel Pathway", "Electricity Demand [kWh / kg fuel]", "Lignocellulosic Biomass Demand [kg / kg fuel]", "NG Demand [GJ / kg fuel]", "Water Demand [m^3 / kg fuel]", "CO2 Demand [kg CO2 / kg fuel]"
        ]

        # Create a DataFrame for the output data
        resource_df = pd.DataFrame(output_resource_data, columns=output_resource_columns)

        # Write the output data to a CSV file
        output_resource_file = f"{fuel}_resource_demands.csv"
        resource_df.to_csv(os.path.join(output_dir_production, output_resource_file), index=False)
        print(f"Output CSV file created: {os.path.join(output_dir_production, output_resource_file)}")

        # Define the output CSV column names
        output_columns = [
            "Fuel", "Hydrogen Source", "Carbon Source", "Electricity Source", "Pathway Name", "Region", "Number", "Year",
            "CapEx [$/tonne]", "OpEx [$/tonne]", "LCOF [$/tonne]", "Emissions [kg CO2e / kg fuel]", "Comment"
        ]

        # Create a DataFrame for the output data
        output_df = pd.DataFrame(output_data, columns=output_columns)

        # Write the output data to a CSV file
        output_file = f"{fuel}_costs_emissions.csv"
        output_df.to_csv(os.path.join(output_dir_production, output_file), index=False)

        print(f"Output CSV file created: {os.path.join(output_dir_production, output_file)}")


    # Gate to Pump Processes
    processes = sorted(set(
        proc.strip()
        for procs in pathway_df['GTT processes'].dropna()
        for proc in str(procs).split(",")
        if proc.strip()
    ))

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
                region,instal_factor_low,instal_factor_high,src,water_price,src,NG_price,src,NG_fugitive_emissions,src,LCB_price,src,LCB_upstream_emissions,src,grid_price,src,grid_emissions_intensity,src,solar_price,src,solar_emissions_intensity,src,wind_price,src,wind_emissions_intensity,src,nuke_price,src,nuke_emissions_intensity,src,hourly_labor_rate,src = row
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
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_hydrogen("LTE",instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod = calculate_production_costs_emissions_STP_hydrogen("LTE",instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx -= CapEx_H2prod
                    OpEx -= OpEx_H2prod
                    emissions -= emissions_H2prod
                    fuel = "liquid_hydrogen"
                    comment = "Liquefaction of STP hydrogen to cryogenic hydrogen at atmospheric pressure"
                elif process == "hydrogen_compression":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_compressed_hydrogen("LTE",instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod = calculate_production_costs_emissions_STP_hydrogen("LTE",instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx -= CapEx_H2prod
                    OpEx -= OpEx_H2prod
                    emissions -= emissions_H2prod
                    fuel = "compressed_hydrogen"
                    comment = "compression of STP hydrogen to gaseous hydrogen at 700 bar"
                elif process == "hydrogen_to_ammonia_conversion":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_ammonia("LTE",instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod = calculate_production_costs_emissions_STP_hydrogen("LTE",instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx -= CapEx_H2prod*NH3_H2_demand
                    OpEx -= OpEx_H2prod*NH3_H2_demand
                    emissions -= emissions_H2prod*NH3_H2_demand
                    fuel = "ammonia"
                    comment = "conversion of STP hydrogen to liquid cryogenic ammonia at atmospheric pressure"
                elif process == "ng_liquefaction":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_NG(instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity)
                    CapEx_NGprod, OpEx_NGprod, emissions_NGprod = calculate_production_costs_emissions_NG(water_price,NG_price,elect_price,elect_emissions_intensity)
                    CapEx -= CapEx_NGprod
                    OpEx -= OpEx_NGprod
                    emissions -= emissions_NGprod
                    fuel = "ng"
                    comment = "conversion of STP natural gas to liquid cryogenic natural gas at atmospheric pressure"
                CapEx *= 1000 # convert to $/tonne
                OpEx *= 1000 # convert to $/tonne
                LCOF = CapEx + OpEx # in $/tonne
                calculated_row = [fuel, H_pathway, C_pathway, E_pathway, process_pathway, region, 1, 2024, CapEx, OpEx, LCOF, emissions, comment]
                output_data.append(calculated_row)

        # Create a DataFrame for the output data
        output_df = pd.DataFrame(output_data, columns=output_columns)

        # Write the output data to a CSV file
        output_file = f"{process}_costs_emissions.csv"
        output_df.to_csv(os.path.join(output_dir_process, output_file), index=False)

        print(f"Output CSV file created: {os.path.join(output_dir_process, output_file)}")


if __name__ == "__main__":
    main()
