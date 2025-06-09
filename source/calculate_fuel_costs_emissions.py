"""
Date: Aug 21, 2024
Purpose: Prepare .csv files contained in input_fuel_pathway_data using consistent assumptions.
"""

import pandas as pd
import os
from common_tools import get_top_dir, ensure_directory_exists
from load_inputs import load_molecular_info, load_technology_info, load_global_parameters

top_dir = get_top_dir()

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
MW_FTdiesel = molecular_info["MW_FTdiesel"]["value"]
nC_FTdiesel = molecular_info["nC_FTdiesel"]["value"]
####################################################################################################################

############################## Read in technology info and calculate derived parameters ############################
tech_info = load_technology_info()

# Inputs for STP H2 production from ATR with 99% CO2 capture rate from Zang et al 2024 (ATR-CC-R-OC case) #
ATRCCS_prod = tech_info["H2_ATRCCS"]["hourly_prod"]["value"]
H2_ATRCCS_elect_demand = tech_info["H2_ATRCCS"]["elec_cons"]["value"]/ATRCCS_prod # [kWh elect/kg H2] from Zang et al 2024
H2_ATRCCS_NG_demand = tech_info["H2_ATRCCS"]["NG_cons"]["value"] / ATRCCS_prod # [GJ NG/kg H2] from Zang et al 2024
H2_ATRCCS_NG_demand = H2_ATRCCS_NG_demand * (1 + NG_NG_demand_kg)     # Also account for the additional NG consumed to process and recover the NG, from GREET 2024
H2_ATRCCS_water_demand = tech_info["H2_ATRCCS"]["water_cons"]["value"] / ATRCCS_prod # [m^3 H2O/kg H2] from Zang et al 2024
ATRCCS_TPC = global_parameters["2019_to_2024_USD"]["value"] * tech_info["H2_ATRCCS"]["TPC_2019"]["value"]
H2_ATRCCS_base_CapEx = ATRCCS_TPC / 365 * tech_info["H2_ATRCCS"]["CRF"]["value"] # From Zang et al 2024 and H2A
H2_ATRCCS_onsite_emissions = tech_info["H2_ATRCCS"]["emissions"]["value"] / ATRCCS_prod # [kg CO2e output/kg H2] from Zang et al 2024
H2_ATRCCS_yearly_output = 365*24*ATRCCS_prod # [kg H2/year] from Zang et al 2024
###########################################################################################################

### Inputs for STP H2 production from SMR with 96% CO2 capture rate from Zang et al 2024 (SMR-CCS case) ###
SMRCCS_prod = tech_info["H2_SMRCCS"]["hourly_prod"]["value"]
H2_SMRCCS_elect_demand = tech_info["H2_SMRCCS"]["elec_cons"]["value"] / SMRCCS_prod # [kWh elect/kg H2] from Zang et al 2024
H2_SMRCCS_NG_demand = tech_info["H2_SMRCCS"]["NG_cons"]["value"] / SMRCCS_prod # [GJ NG/kg H2] from Zang et al 2024
H2_SMRCCS_NG_demand = H2_SMRCCS_NG_demand * (1 + NG_NG_demand_kg)     # Also account for the additional NG consumed to process and recover the NG, from GREET 2024
H2_SMRCCS_water_demand = tech_info["H2_SMRCCS"]["water_cons"]["value"] / SMRCCS_prod # [m^3 water/kg H2] from Zang et al 2024
H2_SMRCCS_onsite_emissions = tech_info["H2_SMRCCS"]["emissions"]["value"] / SMRCCS_prod # [kg CO2e output/kg H2] from Zang et al 2024
SMRCCS_TPC = global_parameters["2019_to_2024_USD"]["value"] * tech_info["H2_SMRCCS"]["TPC_2019"]["value"]
H2_SMRCCS_base_CapEx = SMRCCS_TPC / 365 * tech_info["H2_SMRCCS"]["CRF"]["value"] # [2024$/kg] amortized TPC from Zang et al 2024
H2_SMRCCS_yearly_output = 365 * 24 * SMRCCS_prod # [kg H2/year] from Zang et al 2024
###########################################################################################################

######## Inputs for STP H2 production from SMR without CO2 capture from Zang et al 2024 (SMR case) ########
SMR_prod = tech_info["H2_SMR"]["hourly_prod"]["value"]
H2_SMR_elect_demand = tech_info["H2_SMR"]["elec_cons"]["value"] / SMR_prod # [kWh elect/kg H2] from Zang et al 2024
SMR_NG = tech_info["H2_SMR"]["NG_cons"]["value"] - tech_info["H2_SMR"]["steam_byproduct"]["value"] / tech_info["H2_SMR"]["boiler_eff"]["value"]    # [GJ NG/hr]: NG consumption including steam displacement at 80% boiler efficiency
H2_SMR_NG_demand = SMR_NG / SMR_prod # [GJ NG/kg H2] from Zang et al 2024
H2_SMR_NG_demand = H2_SMR_NG_demand * (1 + NG_NG_demand_kg)     # Also account for the additional NG consumed to process and recover the NG, from GREET 2024
H2_SMR_water_demand = tech_info["H2_SMR"]["water_cons"]["value"] / SMR_prod # [m^3 water/kg H2] from Zang et al 2024
H2_SMR_onsite_emissions = tech_info["H2_SMR"]["emissions"]["value"] / SMR_prod # [kg CO2e output/kg H2] from Zang et al 2024
SMR_TPC = global_parameters["2019_to_2024_USD"]["value"] * tech_info["H2_SMR"]["TPC_2019"]["value"]
H2_SMR_base_CapEx = SMR_TPC / 365 * tech_info["H2_SMR"]["CRF"]["value"] # [2024$/kg] amortized TPC from Zang et al 2024
H2_SMR_yearly_output = 365 * 24 * SMR_prod # [kg H2/year] from Zang et al 2024
###########################################################################################################

## Inputs for STP H2 production from lignocellulosic biomass (LCB) gasification (BG) without CO2 capture ##
H2_BG_NG_demand = tech_info["H2_BG"]["NG_demand"]["value"]
H2_BG_NG_demand = H2_BG_NG_demand * (1 + NG_NG_demand_kg)     # Also account for the additional NG consumed to process and recover the NG, from GREET 2024
H2_BG_base_CapEx = global_parameters["2015_to_2024_USD"]["value"] * tech_info["H2_BG"]["base_CapEx_2015"]["value"]  # Base CapEx, converted from 2015 USD to 2024 USD
H2_BG_emissions = 1.913592126 # [kg CO2e/bone-dry kg] process emissions from gasification of lignocellulosic biomass (LCB)
H2_BG_onsite_emissions = tech_info["H2_BG"]["onsite_emissions"]["value"] - tech_info["H2_BG"]["LCB_gasification_emissions"]["value"] * tech_info["H2_BG"]["LCB_demand"]["value"] # [kg CO2e / kg H2] from H2A #NOTE: includes biogenic credit
###########################################################################################################

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
NH3_H2_demand = 3/2 * molecular_info["MW_H2"]["value"] / molecular_info["MW_NH3"]["value"] # [kg H2/kg NH3] stoichiometry
NH3_elect_demand = tech_info["NH3"]["elect_demand"]["value"] - tech_info["H2_LTE"]["elect_demand"]["value"]*NH3_H2_demand # subtract electrical demand from LTE H2 process
NH3_water_demand = tech_info["NH3"]["water_demand_LTE"]["value"] - tech_info["H2_LTE"]["water_demand"]["value"]*NH3_H2_demand # subtract water demand from LTE H2 process
NH3_base_CapEx = tech_info["NH3"]["base_CapEx_LTE"]["value"] - tech_info["H2_LTE"]["base_CapEx"]["value"]*NH3_H2_demand # subtract base CapEx from LTE H2 process
NH3_full_time_employed = 10 # [full-time employees] from H2A
NH3_full_time_employed = tech_info["NH3"]["employees_LTE"]["value"] - tech_info["H2_LTE"]["employees"]["value"] # subtract employees from LTE H2 process
###########################################################################################################

###########################################################################################################

####################################################################################################################

def calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    if H_pathway == "LTE":
        elect_demand = tech_info["H2_LTE"]["elect_demand"]["value"]
        LCB_demand = tech_info["H2_LTE"]["LCB_demand"]["value"]
        NG_demand = tech_info["H2_LTE"]["NG_demand"]["value"]
        water_demand = tech_info["H2_LTE"]["water_demand"]["value"]
        base_CapEx = tech_info["H2_LTE"]["base_CapEx"]["value"]
        full_time_employed = tech_info["H2_LTE"]["employees"]["value"]
        yearly_output = tech_info["H2_LTE"]["yearly_output"]["value"]
        onsite_emissions = tech_info["H2_LTE"]["onsite_emissions"]["value"]
    elif H_pathway == "ATRCCS":
        elect_demand = H2_ATRCCS_elect_demand
        LCB_demand = tech_info["H2_ATRCCS"]["LCB_demand"]["value"]
        NG_demand = H2_ATRCCS_NG_demand
        water_demand = H2_ATRCCS_water_demand
        base_CapEx = H2_ATRCCS_base_CapEx
        full_time_employed = tech_info["H2_ATRCCS"]["employees"]["value"]
        yearly_output = H2_ATRCCS_yearly_output
        onsite_emissions = H2_ATRCCS_onsite_emissions
    elif H_pathway == "SMRCCS":
        elect_demand = H2_SMRCCS_elect_demand
        LCB_demand = tech_info["H2_SMRCCS"]["LCB_demand"]["value"]
        NG_demand = H2_SMRCCS_NG_demand
        water_demand = H2_SMRCCS_water_demand
        base_CapEx = H2_SMRCCS_base_CapEx
        full_time_employed = tech_info["H2_SMRCCS"]["employees"]["value"]
        yearly_output = H2_SMRCCS_yearly_output
        onsite_emissions = H2_SMRCCS_onsite_emissions
    elif H_pathway == "SMR":
        elect_demand = H2_SMR_elect_demand
        LCB_demand = tech_info["H2_SMR"]["LCB_demand"]["value"]
        NG_demand = H2_SMR_NG_demand
        water_demand = H2_SMR_water_demand
        base_CapEx = H2_SMR_base_CapEx
        full_time_employed = tech_info["H2_SMR"]["employees"]["value"]
        yearly_output = H2_SMR_yearly_output
        onsite_emissions = H2_SMR_onsite_emissions
    elif H_pathway == "BG":
        elect_demand = tech_info["H2_BG"]["elec_demand"]["value"]
        LCB_demand = tech_info["H2_BG"]["LCB_demand"]["value"]
        NG_demand = H2_BG_NG_demand
        water_demand = tech_info["H2_BG"]["water_demand"]["value"]
        base_CapEx = H2_BG_base_CapEx
        full_time_employed = tech_info["H2_BG"]["employees"]["value"]
        yearly_output = tech_info["H2_BG"]["yearly_output"]["value"]
        onsite_emissions = H2_BG_onsite_emissions
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    LCB_OpEx = LCB_demand*LCB_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx + LCB_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_CH4_leakage + LCB_demand*LCB_upstream_emissions + onsite_emissions # note fugitive emissions given as fraction
    
    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_liquid_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    base_CapEx = tech_info["H2_liquefaction"]["base_CapEx"]["value"]
    elect_demand = tech_info["H2_liquefaction"]["elec_demand"]["value"]
    # calculate liquefaction values
    CapEx = base_CapEx*instal_factor
    OpEx = (op_maint_rate + tax_rate)*CapEx + elect_demand*elect_price
    emissions = elect_demand*elect_emissions_intensity
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx
    OpEx += H2_OpEx
    emissions += H2_emissions

    return CapEx, OpEx, emissions
    
def calculate_production_costs_emissions_NG(water_price,NG_price,elect_price,elect_emissions_intensity):
    NG_demand = NG_NG_demand_GJ     # Natural gas consumed in the recovery and processing stages, in GJ NG consumed / kg NG produced
    water_demand = NG_water_demand  # Water consumed in the recovery and processing stages, in m^3 water consumed / kg NG produced
    elect_demand = NG_elect_demand  # Electricity consumed in the recovery and processing stages, in kWh electricity consumed / kg NG produced
    # Assign the price of NG to the OpEx
    CapEx = 0
    OpEx = NG_price * (NG_HHV + NG_NG_demand_GJ) + water_price * water_demand + elect_price * elect_demand   # Account for both the NG produced and the NG consumed in the recovery and processing stages
    emissions = NG_CO2_emissions + NG_CH4_leakage * NG_GWP + elect_demand * elect_emissions_intensity
    
    return CapEx, OpEx, emissions
    
def calculate_production_costs_emissions_liquid_NG(instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity):
    base_CapEx = NG_liq_base_CapEx
    NG_demand = NG_liq_NG_demand_GJ    # Natural gas consumed to power the liquefaction process, in GJ/kg
    water_demand = NG_liq_water_demand  # Water consumed during the liquefaction process, in m^3/kg
    elect_demand = NG_liq_elect_demand  # Electricity consumed during the liquefaction process, in kWh/kg
    
    # calculate liquefaction values
    CapEx = base_CapEx*instal_factor
    OpEx = (op_maint_rate + tax_rate)*CapEx + NG_demand*NG_price + water_demand*water_price + elect_demand*elect_price
    emissions = NG_liq_CO2_emissions + NG_liq_CH4_leakage * NG_GWP + elect_demand * elect_emissions_intensity  # kg CO2e / kg NG
    
    # add H2 feedstock cost and emissions
    NG_CapEx, NG_OpEx, NG_emissions = calculate_production_costs_emissions_NG(water_price,NG_price,elect_price,elect_emissions_intensity)
    CapEx += NG_CapEx
    OpEx += NG_OpEx
    emissions += NG_emissions

    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_compressed_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    base_CapEx = tech_info["H2_compression"]["base_CapEx"]["value"]
    elect_demand = tech_info["H2_compression"]["elec_demand"]["value"]
    # calculate compression values
    CapEx = base_CapEx*instal_factor
    OpEx = (op_maint_rate + tax_rate)*CapEx + elect_demand*elect_price
    emissions = elect_demand*elect_emissions_intensity
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx
    OpEx += H2_OpEx
    emissions += H2_emissions

    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_ammonia(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    elect_demand = NH3_elect_demand
    H2_demand = NH3_H2_demand
    NG_demand = tech_info["NH3"]["NG_demand"]["value"]
    water_demand = NH3_water_demand
    base_CapEx = NH3_base_CapEx
    full_time_employed = NH3_full_time_employed
    yearly_output = tech_info["NH3"]["yearly_output"]["value"]
    onsite_emissions = tech_info["NH3"]["onsite_emissions"]["value"]
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_CH4_leakage + onsite_emissions # note fugitive emissions given as fraction
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx*H2_demand
    OpEx += H2_OpEx*H2_demand
    emissions += H2_emissions*H2_demand

    return CapEx, OpEx, emissions


def calculate_production_costs_emissions_methanol(H_pathway,C_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    elect_demand = tech_info["MeOH"]["elect_demand"]["value"]
    H2_demand = tech_info["MeOH"]["H2_demand"]["value"]
    CO2_demand = tech_info["MeOH"]["CO2_demand"]["value"]
    NG_demand = tech_info["MeOH"]["NG_demand"]["value"]
    water_demand = tech_info["MeOH"]["water_demand"]["value"]
    base_CapEx = tech_info["MeOH"]["base_CapEx"]["value"]
    full_time_employed = tech_info["MeOH"]["employees"]["value"]
    yearly_output = tech_info["MeOH"]["yearly_output"]["value"]
    onsite_emissions = tech_info["MeOH"]["onsite_emissions"]["value"]
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_CH4_leakage # note fugitive emissions given as fraction
    if ((C_pathway != "BEC") & (C_pathway != "DAC")): # ignore onsite emissions if "captured" CO2 (BEC or DAC)
        emissions += onsite_emissions
    # add H2 feedstock costs and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx*H2_demand
    OpEx += H2_OpEx*H2_demand
    emissions += H2_emissions*H2_demand
    # add CO2 feedstock costs and emissions
    if C_pathway == "BEC":
        CO2_CapEx = 0 # No CapEx because we assume BEC CO2 is purchased externally at a fixed price 
        CO2_OpEx = BEC_CO2_price
        CO2_emissions = BEC_CO2_upstream_emissions*CO2_demand - MW_CO2/MW_MeOH # biogenic CO2 credit
    elif C_pathway == "DAC":
        CO2_CapEx = 0 # No CapEx because we assume DAC CO2 is purchased externally at a fixed price 
        CO2_OpEx = DAC_CO2_price
        CO2_emissions = DAC_CO2_upstream_emissions*CO2_demand + DAC_CO2_upstream_NG/NG_HHV*NG_GWP*NG_CH4_leakage*CO2_demand + DAC_CO2_upstream_elect*elect_emissions_intensity*CO2_demand - MW_CO2/MW_MeOH # captured CO2 credit
    elif (C_pathway == "SMRCCS") | (C_pathway == "ATRCCS"):
        CO2_CapEx = 0 # CO2 is "free" after already paying for upstream CCS
        CO2_OpEx = 0 # CO2 is "free" after already paying for upstream CCS
        CO2_emissions = CO2_demand - MW_CO2/MW_MeOH # we are working with already-captured fossil CO2, but not at 100% conversion.
    elif C_pathway == "SMR":
        CO2_CapEx = 0 # assumes integrated plant with syngas conversion
        CO2_OpEx = 0 # assumes integrated plant with syngas conversion
        CO2_emissions = -MW_CO2/MW_MeOH # fossil CO2 that would have been emitted by SMR is instead (temporarily) embodied in fuel 
    elif C_pathway == "BG":
        CO2_CapEx = 0 # assumes integrated plant with conversion of gasified biomass
        CO2_OpEx = 0 # assumes integrated plant with conversion of gasified biomass
        CO2_emissions = 0 # biogenic CO2 credit is already applied
    CapEx += CO2_CapEx*CO2_demand
    OpEx += CO2_OpEx*CO2_demand
    emissions += CO2_emissions 

    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_FTdiesel(H_pathway,C_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    elect_demand = tech_info["FTdiesel"]["elect_demand"]["value"]
    H2_demand = tech_info["FTdiesel"]["H2_demand"]["value"]
    CO2_demand = tech_info["FTdiesel"]["CO2_demand"]["value"]
    NG_demand = tech_info["FTdiesel"]["NG_demand"]["value"]
    water_demand = tech_info["FTdiesel"]["water_demand"]["value"]
    base_CapEx = tech_info["FTdiesel"]["base_CapEx"]["value"]
    full_time_employed = tech_info["FTdiesel"]["employees"]["value"]
    yearly_output = tech_info["FTdiesel"]["yearly_output"]["value"]
    onsite_emissions = tech_info["FTdiesel"]["onsite_emissions"]["value"]
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_CH4_leakage # note fugitive emissions given as fraction
    if ((C_pathway != "BEC") & (C_pathway != "DAC")): # ignore onsite emissions if "captured" CO2 (BEC or DAC)
        emissions += onsite_emissions
    # add H2 feedstock costs and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx*H2_demand
    OpEx += H2_OpEx*H2_demand
    emissions += H2_emissions*H2_demand
    # add CO2 feedstock costs and emissions
    if C_pathway == "BEC":
        CO2_CapEx = 0 # No CapEx because we assume BEC CO2 is purchased externally at a fixed price 
        CO2_OpEx = BEC_CO2_price
        CO2_emissions = BEC_CO2_upstream_emissions*CO2_demand - nC_FTdiesel*MW_CO2/MW_FTdiesel # biogenic CO2 credit
    elif C_pathway == "DAC":
        CO2_CapEx = 0 # No CapEx because we assume DAC CO2 is purchased externally at a fixed price 
        CO2_OpEx = DAC_CO2_price
        CO2_emissions = DAC_CO2_upstream_emissions*CO2_demand + DAC_CO2_upstream_NG/NG_HHV*NG_GWP*NG_CH4_leakage*CO2_demand + DAC_CO2_upstream_elect*elect_emissions_intensity*CO2_demand - nC_FTdiesel*MW_CO2/MW_FTdiesel # captured CO2 credit
    elif (C_pathway == "SMRCCS") | (C_pathway == "ATRCCS"):
        CO2_CapEx = 0 # CO2 is "free" after already paying for upstream CCS
        CO2_OpEx = 0 # CO2 is "free" after already paying for upstream CCS
        CO2_emissions = CO2_demand - nC_FTdiesel*MW_CO2/MW_FTdiesel # we are working with already-captured fossil CO2, but not with 100% conversion.
    elif C_pathway == "SMR":
        CO2_CapEx = 0 # assumes integrated plant with syngas conversion
        CO2_OpEx = 0 # assumes integrated plant with syngas conversion
        CO2_emissions = -nC_FTdiesel*MW_CO2/MW_FTdiesel # fossil CO2 that would have been emitted by SMR is instead embodied in fuel 
    elif C_pathway == "BG":
        CO2_CapEx = 0 # assumes integrated plant with conversion of gasified biomass
        CO2_OpEx = 0 # assumes integrated plant with conversion of gasified biomass
        CO2_emissions = 0 # biogenic CO2 credit is already applied
    CapEx += CO2_CapEx*CO2_demand
    OpEx += CO2_OpEx*CO2_demand
    emissions += CO2_emissions 

    return CapEx, OpEx, emissions


# Added by GE for 10/18 
def calculate_resource_demands_STP_hydrogen(H_pathway):
    if H_pathway == "LTE":
        elect_demand = tech_info["H2_LTE"]["elect_demand"]["value"]
        LCB_demand = tech_info["H2_LTE"]["LCB_demand"]["value"]
        NG_demand = tech_info["H2_LTE"]["NG_demand"]["value"]
        water_demand = tech_info["H2_LTE"]["water_demand"]["value"]
        CO2_demand = 0
    elif H_pathway == "ATRCCS":
        elect_demand = H2_ATRCCS_elect_demand
        LCB_demand = tech_info["H2_ATRCCS"]["LCB_demand"]["value"]
        NG_demand = H2_ATRCCS_NG_demand
        water_demand = H2_ATRCCS_water_demand
        CO2_demand = 0
    elif H_pathway == "SMRCCS":
        elect_demand = H2_SMRCCS_elect_demand
        LCB_demand = tech_info["H2_SMRCCS"]["LCB_demand"]["value"]
        NG_demand = H2_SMRCCS_NG_demand
        water_demand = H2_SMRCCS_water_demand
        CO2_demand = 0
    elif H_pathway == "SMR":
        elect_demand = H2_SMR_elect_demand
        LCB_demand = tech_info["H2_SMR"]["LCB_demand"]["value"]
        NG_demand = H2_SMR_NG_demand
        water_demand = H2_SMR_water_demand
        CO2_demand = 0
    elif H_pathway == "BG":
        elect_demand = tech_info["H2_BG"]["elec_demand"]["value"]
        LCB_demand = tech_info["H2_BG"]["LCB_demand"]["value"]
        NG_demand = H2_BG_NG_demand
        water_demand = tech_info["H2_BG"]["water_demand"]["value"]
        CO2_demand = 0

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


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


# end of section added by Grace for 10/18

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
    fuels = pathway_df['WTG fuels'].dropna().tolist()
    fuel_contents = pathway_df['fuel contents'].dropna().tolist()
    processes = pathway_df['GTT processes'].dropna().tolist()
    Esources = pathway_df['electricity sources'].dropna().tolist()
    Hsources = pathway_df['hydrogen sources'].dropna().tolist()
    Csources = pathway_df['carbon sources'].dropna().tolist()
    # Well to Gate fuel production
    for fuel in fuels:
        if "fossil" in fuel_contents[fuels.index(fuel)]:
            fuel_pathways=["fossil"]
            H_pathways = ["n/a"]
            C_pathways = ["n/a"]
            E_pathways = ["n/a"]
        else:
            if "C" in fuel_contents[fuels.index(fuel)]: # if fuel contains carbon
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
        
        # First, handle fossil production pathway
            
        for row_index, row in input_df.iterrows():
            region,instal_factor_low,instal_factor_high,src,water_price,src,NG_price,src,NG_fugitive_emissions,src,LCB_price,src,LCB_upstream_emissions,src,grid_price,src,grid_emissions_intensity,src,solar_price,src,solar_emissions_intensity,src,wind_price,src,wind_emissions_intensity,src,nuke_price,src,nuke_emissions_intensity,src,hourly_labor_rate,src = row
            
            # Calculate the average installation factor
            instal_factor = (instal_factor_low + instal_factor_high) / 2
            
            if fuel == "ng":
                CapEx, OpEx, emissions = calculate_production_costs_emissions_NG(water_price, NG_price,elect_price,elect_emissions_intensity)
                comment = "natural gas at standard temperature and pressure"
            elif fuel == "lng":
                CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_NG(instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity)
                comment = "liquid natural gas at atmospheric pressure"
        
        # Next, iterate through the non-fossil production pathways
        for fuel_pathway in fuel_pathways:
            pathway_index = fuel_pathways.index(fuel_pathway)
            H_pathway = H_pathways[pathway_index]
            C_pathway = C_pathways[pathway_index]

            if fuel == "hydrogen":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
                comment = "hydrogen at standard temperature and pressure"
            elif fuel == "liquid_hydrogen":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_liquid_hydrogen(H_pathway)
                comment = "Liquid cryogenic hydrogen at atmospheric pressure"
            elif fuel == "compressed_hydrogen":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_compressed_hydrogen(H_pathway)
                comment = "compressed gaseous hydrogen at 700 bar"
            elif fuel == "ng":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_NG()
                comment = "natural gas at standard temperature and pressure"
            elif fuel == "lng":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_liquid_NG()
                comment = "liquid natural gas at atmospheric pressure"
            elif fuel == "ammonia":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_ammonia(H_pathway)
                comment = "Liquid cryogenic ammonia at atmospheric pressure"
            elif fuel == "methanol":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_methanol(H_pathway, C_pathway)
                comment = "Liquid methanol at STP"
            elif fuel == "FTdiesel":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_FTdiesel(H_pathway, C_pathway)
                comment = "liquid Fischer--Tropsch diesel fuel at STP"

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
                if fuel == "hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "hydrogen at standard temperature and pressure"
                elif fuel == "liquid_hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid cryogenic hydrogen at atmospheric pressure"
                elif fuel == "compressed_hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_compressed_hydrogen(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "compressed gaseous hydrogen at 700 bar"
                elif fuel == "ammonia":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_ammonia(H_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid cryogenic ammonia at atmospheric pressure"
                elif fuel == "ng":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_NG(water_price, NG_price, elect_price,elect_emissions_intensity)
                    comment = "natural gas at standard temperature and pressure"
                elif fuel == "lng":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_NG(instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity)
                    comment = "liquid natural gas at atmospheric pressure"
                elif fuel == "methanol":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_methanol(H_pathway,C_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "liquid methanol at STP"
                elif fuel == "FTdiesel":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_FTdiesel(H_pathway,C_pathway,instal_factor,water_price,NG_price,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "liquid Fischer--Tropsch diesel fuel at STP"
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
    processes = ["hydrogen_liquefaction", "hydrogen_compression", "hydrogen_to_ammonia_conversion", "ng_liquefaction"]
    
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
