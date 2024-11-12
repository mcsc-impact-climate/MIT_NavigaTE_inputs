"""
Date: Aug 21, 2024
Purpose: Prepare .csv files contained in input_fuel_pathway_data using consistent assumptions.
"""

import pandas as pd
import os
from common_tools import get_top_dir, ensure_directory_exists

# Function to calculate CapEx, OpEx, LCOF, and production GHG emissions for STP hydrogen
workhours_per_year = 52*40 # number of work-hours per year
NG_HHV = 0.0522 # GJ/kg NG
NG_GWP = 28 # GWP100 of methane (surrogate for NG) using IPCC-AR5 as in MEPC.391(81)
gen_admin_rate = 0.2 # 20% G&A rate
op_maint_rate = 0.04 # O&M rate
tax_rate = 0.02 # 2% tax rate
NG_percent_fugitive = 0.02 # Assume 2% fugitive NG emissions
BEC_CO2_price = 0.04 # [2024$/kg CO2] price of biogenic CO2 (e.g. ~$40/tonne from biomass BEC, ~$90/tonne from biogas BEC, ~$200+/tonne from DAC) #NOTE: just a placeholder value for now # This input should probably be regionalized or made dependent on LCB price
BEC_CO2_upstream_emissions = 0.02 # [kg CO2e/kg CO2] upstream emissions from bioenergy plant with CO2 capture (e.g. 0.02 from biomass BEC, 0.05 from biogas BEC..) #NOTE: just a placeholder value for now
DAC_CO2_price = 0.2 # [2024$/kg CO2] price of captured CO2 (e.g. ~$40/tonne from biomass BEC, ~$90/tonne from biogas BEC, ~$200+/tonne from DAC) #NOTE: just a placeholder value for now
DAC_CO2_upstream_emissions = 0.02 # [kg CO2e/kg CO2] upstream emissions from direct-air CO2 capture (e.g. 0.02 from biomass BEC, 0.05 from biogas BEC..) #NOTE: just a placeholder value for now
MW_CO2 = 44.01 # [g/mol] avg molecular weight of carbon dioxide 
MW_MeOH = 32.04 # [g/mol] avg molecular weight of methanol
MW_NH3 = 17.03 # [g/mol] avg molecular weight of ammonia
MW_H2 = 2.016 # [g/mol] avg molecular weight of hydrogen
MW_FTdiesel = 167.3 # [g/mol] avg molecular weight of FT diesel
nC_FTdiesel = 12 # [-] avg number of carbon atoms per molecular of FT diesel

# Inputs for STP H2 production from low-temperature water electrolysis
H2_LTE_elect_demand = 51.03004082 # [kWh elect/kg H2] from Aspen Plus
H2_LTE_LCB_demand = 0 # [kg/kg H2]
H2_LTE_NG_demand = 0 # [GJ NG/kg H2] zero for LTE
H2_LTE_water_demand = 0.01430886 # [m^3 H2O/kg H2] from Aspen Plus
H2_LTE_base_CapEx = 0.56079 # [2024$/kg H2] from H2A
H2_LTE_full_time_employed = 10 # [full-time employees] from H2A
H2_LTE_yearly_output = 20003825 # [kg H2/year] from Aspen Plus
H2_LTE_onsite_emissions = 0 # [kg CO2e/kg H2] zero for LTE

# Inputs for STP H2 production from autothermal reforming with 99% CO2 capture rate from Zang et al 2021 (ATR-CC-R-OC case)
ATRCCS_prod = 20125 # [kg H2/hr] Table 6, column 5: hourly H2 production
ATRCCS_elec = 58000 # [kW elec] Table 6, column 5: electricity consumption
ATRCCS_NG = 3376 # [GJ NG/hr] Table 6, column 5: hourly NG consumption
ATRCCS_water = 96.831 # [m^3/hr] Table S5: water flow 19
ATRCCS_emissions = 221 # [kg CO2e/hr] Table 6, column 5: hourly CO2e emissions
ATRCCS_TPC = 1.2253*1150 # [2024$/(kg H2/day)] total plant cost from Fig 4 case 5 TPC (inflation adjusted from 2019 USD to 2024 USD)
ATRCCS_CRF = 0.1018522 # [-] capital recovery factor from H2A based on data from Table 5
H2_ATRCCS_elect_demand = ATRCCS_elec/ATRCCS_prod # [kWh elect/kg H2] from Zang et al 2021
H2_ATRCCS_LCB_demand = 0 # [kg/kg H2]
H2_ATRCCS_NG_demand = ATRCCS_NG/ATRCCS_prod # [GJ NG/kg H2] from Zang et al 2021
H2_ATRCCS_water_demand = ATRCCS_water/ATRCCS_prod # [m^3 H2O/kg H2] from Zang et al 2021
H2_ATRCCS_base_CapEx = ATRCCS_TPC/365*ATRCCS_CRF # [2024$/kg] amortized TPC from Zang et al 2021
H2_ATRCCS_full_time_employed = 22 # [full-time employees] from H2A
H2_ATRCCS_yearly_output = 365*24*ATRCCS_prod # [kg H2/year] from Zang et al 2021
H2_ATRCCS_onsite_emissions = ATRCCS_emissions/ATRCCS_prod # [kg CO2e output/kg H2] from Zang et al 2021

# Inputs for STP H2 production from steam methane reforming with 96% CO2 capture rate from Zang et al 2021 (SMR-CCS case)
SMRCCS_prod = 20125 # [kg H2/hr] Table 6, column 2: hourly H2 production
SMRCCS_elec = 41000 # [kW elec] Table 6, column 2: electricity consumption
SMRCCS_NG = 3947 # [GJ NG/hr] Table 6, column 2: hourly NG consumption
SMRCCS_water = 308.347 + 177.085 # [m^3/hr] Table S2: water flows 8+17
SMRCCS_emissions = 7656 # [kg CO2e/hr] Table 6, column 2: hourly CO2e emissions
SMRCCS_TPC = 1.2253*1419 # [2024$/(kg H2/day)] total plant cost from Fig 4 case 2 TPC (inflation adjusted from 2019 USD to 2024 USD)
SMRCCS_CRF = 0.1018522 # [-] capital recovery factor from H2A based on data from Table 5
H2_SMRCCS_elect_demand = SMRCCS_elec/SMRCCS_prod # [kWh elect/kg H2] from Zang et al 2021
H2_SMRCCS_LCB_demand = 0 # [kg/kg H2]
H2_SMRCCS_NG_demand = SMRCCS_NG/SMRCCS_prod # [GJ NG/kg H2] from Zang et al 2021
H2_SMRCCS_water_demand = SMRCCS_water/SMRCCS_prod # [m^3 H2O/kg H2] from Zang et al 2021
H2_SMRCCS_base_CapEx = SMRCCS_TPC/365*SMRCCS_CRF # [2024$/kg] amortized TPC from Zang et al 2021
H2_SMRCCS_full_time_employed = 22 # [full-time employees] from H2A
H2_SMRCCS_yearly_output = 365*24*SMRCCS_prod # [kg H2/year] from Zang et al 2021
H2_SMRCCS_onsite_emissions = SMRCCS_emissions/SMRCCS_prod # [kg CO2e output/kg H2] from Zang et al 2021

# Inputs for STP H2 production from steam methane reforming without CO2 capture from Zang et al 2021 (SMR case)
SMR_prod = 20126 # [kg H2/hr] Table 6, column 1: hourly H2 production
SMR_elec = 13000 # [kW elec] Table 6, column 1: electricity consumption
SMR_NG = 3712 - 514/0.8 # [GJ NG/hr] (NG consumption including steam displacement at 80% boiler efficiency) Table 6, column 1: hourly NG consumption - Table 6, column 1: by product steam [GJ/hr] / 80%
SMR_water = 336.728 # [m^3/hr] Table S1: water flow 13
SMR_emissions = 188221 # [kg CO2e/hr] Table 6, column 1: hourly CO2e emissions
SMR_TPC = 1.2253*564 # [2024$/(kg H2/day)] total plant cost from Fig 4 case 1 TPC (inflation adjusted from 2019 USD to 2024 USD)
SMR_CRF = 0.1018522 # [-] capital recovery factor from H2A based on data from Table 5
H2_SMR_elect_demand = SMR_elec/SMR_prod # [kWh elect/kg H2] from Zang et al 2021
H2_SMR_LCB_demand = 0 # [kg/kg H2]
H2_SMR_NG_demand = SMR_NG/SMR_prod # [GJ NG/kg H2] from Zang et al 2021
H2_SMR_water_demand = SMR_water/SMR_prod # [m^3 H2O/kg H2] from Zang et al 2021
H2_SMR_base_CapEx = SMR_TPC/365*SMR_CRF # [2024$/kg] amortized TPC from Zang et al 2021
H2_SMR_full_time_employed = 22 # [full-time employees] from H2A
H2_SMR_yearly_output = 365*24*SMR_prod # [kg H2/year] from Zang et al 2021
H2_SMR_onsite_emissions = SMR_emissions/SMR_prod # [kg CO2e output/kg H2] from Zang et al 2021

# Inputs for STP H2 production from lignocellulosic biomass (LCB) gasification (BG) without CO2 capture
H2_BG_elect_demand = 0.98 # [kWh elect/kg H2] from H2A
H2_BG_LCB_demand = 13.49 # [kg/kg H2] from H2A
H2_BG_NG_demand = 0.0062245 # [GJ NG/kg H2] from H2A
H2_BG_water_demand = 0.005 # [m^3 H2O/kg H2] from H2A
H2_BG_base_CapEx = 1.3137*0.32 # [2024$/kg] from H2A
H2_BG_full_time_employed = 54 # [full-time employees] from H2A
H2_BG_yearly_output = 50995026 # [kg H2/year] from H2A
H2_BG_emissions = 1.913592126 # [kg CO2e/bone-dry kg] process emissions from gasification of lignocellulosic biomass (LCB) 
H2_BG_onsite_emissions = 26.16 - H2_BG_emissions*H2_BG_LCB_demand # [kg CO2e/kgH2] from H2A #NOTE: includes biogenic credit

# Inputs for production of liquid H2 at 20 K
H2_liq_base_CapEx = 0.59898 # [2024$/kg]
H2_liq_elect_demand = 8.0 # [kWh elect/kg H2]

# Inputs for production of gaseous H2 at 700 bar and 300 K
H2_comp_base_CapEx = 0.17114 # [2024$/kg]
H2_comp_elect_demand = 3.0 # [kWh elect/kg H2]

# Inputs for liquid NH3 production from arbitrary H2 feedstock
NH3_elect_demand = 10.05314189 # [kWh elect/kg NH3] for LTE ammonia process from Aspen Plus
NH3_H2_demand = 3/2*MW_H2/MW_NH3 # [kg H2/kg NH3] stoichiometry
NH3_elect_demand -= H2_LTE_elect_demand*NH3_H2_demand # subtract electrical demand from LTE H2 process
NH3_NG_demand = 0 # [GJ NG/kg H2] from Aspen Plus
NH3_water_demand = 0.00261861625758975 # [m^3 H2O/kg NH3] for LTE ammonia process from Aspen Plus
NH3_water_demand -= H2_LTE_water_demand*NH3_H2_demand # subtract water demand from LTE H2 process
NH3_base_CapEx = 0.193395 # [2024$/kg] from H2A
NH3_base_CapEx -= H2_LTE_base_CapEx*NH3_H2_demand # subtract base CapEx from LTE H2 process
NH3_full_time_employed = 10 # [full-time employees] from H2A
NH3_full_time_employed -= H2_LTE_full_time_employed # subtract employees from LTE H2 process
NH3_yearly_output = 109306554 # [kg NH3/year] from Aspen Plus
NH3_onsite_emissions = 0 # [kg CO2e/kg NH3] zero for Haber-Bosch

# Inputs for MeOH production from arbitrary H and C feedstocks
MeOH_elect_demand = 0.564623278 # [kWh elect/kg MeOH] for MeOH synthesis process from Aspen Plus
MeOH_H2_demand = 0.204464607 # [kg H2/kg MeOH] for MeOH synthesis process from Aspen Plus
MeOH_CO2_demand = 1.662087119 # [kg CO2/kg MeOH] for MeOH synthesis process from Aspen Plus
MeOH_NG_demand = 0 # [GJ NG/kg H2] from Aspen Plus
MeOH_water_demand = 0.000318771 # [m^3 H2O/kg MeOH] for MeOH synthesis process from Aspen Plus
MeOH_base_CapEx = 0.103556 # [2024$/kg] from H2A
MeOH_full_time_employed = 68 # [full-time employees] from H2A
MeOH_yearly_output = 194924166 # [kg MeOH/year] from Aspen Plus
MeOH_onsite_emissions = 0.3153886337 # [kg CO2e/kg MeOH] synthesis process emissions in Aspen Plus

# Inputs for Fischer-Tropsch diesel production from arbitrary H and C feedstocks
FTdiesel_elect_demand = 0.246 # [kWh elect/kg FTdiesel] for FTdiesel synthesis process from Aspen Plus
FTdiesel_H2_demand = 0.635 # [kg H2/kg FTdiesel] for FTdiesel synthesis process from Aspen Plus
FTdiesel_CO2_demand = 6.80 # [kg CO2/kg FTdiesel] for FTdiesel synthesis process from Aspen Plus
FTdiesel_NG_demand = 0 # [GJ NG/kg H2] from Aspen Plus
FTdiesel_water_demand = 0.00166 # [m^3 H2O/kg FTdiesel] for FTdiesel synthesis process from Aspen Plus
FTdiesel_base_CapEx = 0.322 # [2024$/kg] from H2A
FTdiesel_full_time_employed = 80 # [full-time employees] from H2A
FTdiesel_yearly_output = 128202750 # [kg FTdiesel/year] from Aspen Plus
FTdiesel_onsite_emissions = 0.3153886337 # [kg CO2e/kg FTdiesel] synthesis process emissions in Aspen Plus

def calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    if H_pathway == "LTE":
        elect_demand = H2_LTE_elect_demand
        LCB_demand = H2_LTE_LCB_demand
        NG_demand = H2_LTE_NG_demand
        water_demand = H2_LTE_water_demand
        base_CapEx = H2_LTE_base_CapEx
        full_time_employed = H2_LTE_full_time_employed
        yearly_output = H2_LTE_yearly_output
        onsite_emissions = H2_LTE_onsite_emissions
    elif H_pathway == "ATRCCS":
        elect_demand = H2_ATRCCS_elect_demand
        LCB_demand = H2_ATRCCS_LCB_demand
        NG_demand = H2_ATRCCS_NG_demand
        water_demand = H2_ATRCCS_water_demand
        base_CapEx = H2_ATRCCS_base_CapEx
        full_time_employed = H2_ATRCCS_full_time_employed
        yearly_output = H2_ATRCCS_yearly_output
        onsite_emissions = H2_ATRCCS_onsite_emissions
    elif H_pathway == "SMRCCS":
        elect_demand = H2_SMRCCS_elect_demand
        LCB_demand = H2_SMRCCS_LCB_demand
        NG_demand = H2_SMRCCS_NG_demand
        water_demand = H2_SMRCCS_water_demand
        base_CapEx = H2_SMRCCS_base_CapEx
        full_time_employed = H2_SMRCCS_full_time_employed
        yearly_output = H2_SMRCCS_yearly_output
        onsite_emissions = H2_SMRCCS_onsite_emissions
    elif H_pathway == "SMR":
        elect_demand = H2_SMR_elect_demand
        LCB_demand = H2_SMR_LCB_demand
        NG_demand = H2_SMR_NG_demand
        water_demand = H2_SMR_water_demand
        base_CapEx = H2_SMR_base_CapEx
        full_time_employed = H2_SMR_full_time_employed
        yearly_output = H2_SMR_yearly_output
        onsite_emissions = H2_SMR_onsite_emissions
    elif H_pathway == "BG":
        elect_demand = H2_BG_elect_demand
        LCB_demand = H2_BG_LCB_demand
        NG_demand = H2_BG_NG_demand
        water_demand = H2_BG_water_demand
        base_CapEx = H2_BG_base_CapEx
        full_time_employed = H2_BG_full_time_employed
        yearly_output = H2_BG_yearly_output
        onsite_emissions = H2_BG_onsite_emissions
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    LCB_OpEx = LCB_demand*LCB_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx + LCB_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_fugitive_emissions/100 + LCB_demand*LCB_upstream_emissions + onsite_emissions # note fugitive emissions given as percentage
    
    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_liquid_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    base_CapEx = H2_liq_base_CapEx
    elect_demand = H2_liq_elect_demand
    # calculate liquefaction values
    CapEx = base_CapEx*instal_factor
    OpEx = (op_maint_rate + tax_rate)*CapEx + elect_demand*elect_price
    emissions = elect_demand*elect_emissions_intensity
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx
    OpEx += H2_OpEx
    emissions += H2_emissions

    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_compressed_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    base_CapEx = H2_comp_base_CapEx
    elect_demand = H2_comp_elect_demand
    # calculate compression values
    CapEx = base_CapEx*instal_factor
    OpEx = (op_maint_rate + tax_rate)*CapEx + elect_demand*elect_price
    emissions = elect_demand*elect_emissions_intensity
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx
    OpEx += H2_OpEx
    emissions += H2_emissions

    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_ammonia(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    elect_demand = NH3_elect_demand
    H2_demand = NH3_H2_demand
    NG_demand = NH3_NG_demand
    water_demand = NH3_water_demand
    base_CapEx = NH3_base_CapEx
    full_time_employed = NH3_full_time_employed
    yearly_output = NH3_yearly_output
    onsite_emissions = NH3_onsite_emissions
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_fugitive_emissions/100 + onsite_emissions # note fugitive emissions given as percentage
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx*H2_demand
    OpEx += H2_OpEx*H2_demand
    emissions += H2_emissions*H2_demand

    return CapEx, OpEx, emissions


def calculate_production_costs_emissions_methanol(H_pathway,C_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    elect_demand = MeOH_elect_demand
    H2_demand = MeOH_H2_demand
    CO2_demand = MeOH_CO2_demand
    NG_demand = MeOH_NG_demand
    water_demand = MeOH_water_demand
    base_CapEx = MeOH_base_CapEx
    full_time_employed = MeOH_full_time_employed
    yearly_output = MeOH_yearly_output
    onsite_emissions = MeOH_onsite_emissions
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_fugitive_emissions/100 # note fugitive emissions given as percentage
    if ((C_pathway != "BEC") & (C_pathway != "DAC")): # ignore onsite emissions if biogenic CO2
        emissions += onsite_emissions
    # add H2 feedstock costs and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
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
        CO2_emissions = DAC_CO2_upstream_emissions*CO2_demand - MW_CO2/MW_MeOH # captured CO2 credit
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

def calculate_production_costs_emissions_FTdiesel(H_pathway,C_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate):
    elect_demand = FTdiesel_elect_demand
    H2_demand = FTdiesel_H2_demand
    CO2_demand = FTdiesel_CO2_demand
    NG_demand = FTdiesel_NG_demand
    water_demand = FTdiesel_water_demand
    base_CapEx = FTdiesel_base_CapEx
    full_time_employed = FTdiesel_full_time_employed
    yearly_output = FTdiesel_yearly_output
    onsite_emissions = FTdiesel_onsite_emissions
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity + NG_demand/NG_HHV*NG_GWP*NG_fugitive_emissions/100 # note fugitive emissions given as percentage
    if ((C_pathway != "BEC") & (C_pathway != "DAC")): # ignore onsite emissions if biogenic CO2
        emissions += onsite_emissions
    # add H2 feedstock costs and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_factor,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
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
        CO2_emissions = DAC_CO2_upstream_emissions*CO2_demand - nC_FTdiesel*MW_CO2/MW_FTdiesel # captured CO2 credit
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
        elect_demand = H2_LTE_elect_demand
        LCB_demand = H2_LTE_LCB_demand
        NG_demand = H2_LTE_NG_demand
        water_demand = H2_LTE_water_demand
        CO2_demand = 0
    elif H_pathway == "ATRCCS":
        elect_demand = H2_ATRCCS_elect_demand
        LCB_demand = H2_ATRCCS_LCB_demand
        NG_demand = H2_ATRCCS_NG_demand
        water_demand = H2_ATRCCS_water_demand
        CO2_demand = 0
    elif H_pathway == "SMRCCS":
        elect_demand = H2_SMRCCS_elect_demand
        LCB_demand = H2_SMRCCS_LCB_demand
        NG_demand = H2_SMRCCS_NG_demand
        water_demand = H2_SMRCCS_water_demand
        CO2_demand = 0
    elif H_pathway == "SMR":
        elect_demand = H2_SMR_elect_demand
        LCB_demand = H2_SMR_LCB_demand
        NG_demand = H2_SMR_NG_demand
        water_demand = H2_SMR_water_demand
        CO2_demand = 0
    elif H_pathway == "BG":
        elect_demand = H2_BG_elect_demand
        LCB_demand = H2_BG_LCB_demand
        NG_demand = H2_BG_NG_demand
        water_demand = H2_BG_water_demand
        CO2_demand = 0

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_liquid_hydrogen(H_pathway):
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += H2_liq_elect_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_compressed_hydrogen(H_pathway):
    elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += H2_comp_elect_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_ammonia(H_pathway):
    elect_demand = NH3_elect_demand
    LCB_demand = 0
    H2_demand = NH3_H2_demand
    CO2_demand = 0
    NG_demand = NH3_NG_demand
    water_demand = NH3_water_demand

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += H2_elect_demand * H2_demand
    LCB_demand += H2_LCB_demand * H2_demand
    NG_demand += H2_NG_demand * H2_demand
    water_demand += H2_water_demand * H2_demand
    CO2_demand += H2_CO2_demand * H2_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand


def calculate_resource_demands_methanol(H_pathway):
    elect_demand = MeOH_elect_demand
    LCB_demand = 0
    H2_demand = MeOH_H2_demand
    CO2_demand = MeOH_CO2_demand
    NG_demand = MeOH_NG_demand
    water_demand = MeOH_water_demand

    # add H2 resource demands
    H2_elect_demand, H2_LCB_demand, H2_NG_demand, H2_water_demand, H2_CO2_demand = calculate_resource_demands_STP_hydrogen(H_pathway)
    elect_demand += H2_elect_demand * H2_demand
    LCB_demand += H2_LCB_demand * H2_demand
    NG_demand += H2_NG_demand * H2_demand
    water_demand += H2_water_demand * H2_demand
    CO2_demand += H2_CO2_demand * H2_demand

    return elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand

def calculate_resource_demands_FTdiesel(H_pathway):
    elect_demand = FTdiesel_elect_demand
    LCB_demand = 0
    H2_demand = FTdiesel_H2_demand
    CO2_demand = FTdiesel_CO2_demand
    NG_demand = FTdiesel_NG_demand
    water_demand = FTdiesel_water_demand

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
    top_dir = get_top_dir()
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
        # GE - updated outer for loop to calculate and make csv files for resource demands
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
            elif fuel == "ammonia":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_ammonia(H_pathway)
                comment = "Liquid cryogenic ammonia at atmospheric pressure"
            elif fuel == "methanol":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_methanol(H_pathway)
                comment = "Liquid methanol at STP"
            elif fuel == "FTdiesel":
                elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand = calculate_resource_demands_FTdiesel(H_pathway)
                comment = "liquid Fischer--Tropsch diesel fuel at STP"

            calculated_resource_row = [fuel, H_pathway, C_pathway, fuel_pathway, elect_demand, LCB_demand, NG_demand, water_demand, CO2_demand]
            output_resource_data.append(calculated_resource_row)

            for row_index, row in input_df.iterrows():
                region,instal_cost,src,water_price,src,NG_price,src,NG_fugitive_emissions,src,LCB_price,src,LCB_upstream_emissions,src,grid_price,src,grid_emissions_intensity,src,renew_price,src,renew_emissions_intensity,src,nuke_price,src,nuke_emissions_intensity,src,hourly_labor_rate,src = row
                H_pathway = H_pathways[pathway_index]
                C_pathway = C_pathways[pathway_index]
                E_pathway = E_pathways[pathway_index]
                # Check whether we are working with grid or renewable electricity
                # if renewables, fix electricity price and emissions to imposed values
                if E_pathway == "grid":
                    elect_price = grid_price
                    elect_emissions_intensity = grid_emissions_intensity
                elif E_pathway == "renewable":
                    elect_price = renew_price
                    elect_emissions_intensity = renew_emissions_intensity
                elif E_pathway == "nuke":
                    elect_price = nuke_price
                    elect_emissions_intensity = nuke_emissions_intensity
                if fuel == "hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_STP_hydrogen(H_pathway,instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "hydrogen at standard temperature and pressure"
                elif fuel == "liquid_hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_hydrogen(H_pathway,instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid cryogenic hydrogen at atmospheric pressure"
                elif fuel == "compressed_hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_compressed_hydrogen(H_pathway,instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "compressed gaseous hydrogen at 700 bar"
                elif fuel == "ammonia":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_ammonia(H_pathway,instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid cryogenic ammonia at atmospheric pressure"
                elif fuel == "methanol":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_methanol(H_pathway,C_pathway,instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid methanol at STP"
                elif fuel == "FTdiesel":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_FTdiesel(H_pathway,C_pathway,instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
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
    processes = ["hydrogen_liquefaction", "hydrogen_compression", "hydrogen_to_ammonia_conversion"]
    process_pathways = Esources
    E_pathways = Esources
    for process in processes: 
        # List to hold all rows for the output CSV
        output_data = []

        # Iterate through each row in the input data and perform calculations
        for process_pathway in process_pathways:
            pathway_index = process_pathways.index(process_pathway)
            for row_index, row in input_df.iterrows():
                region,instal_cost,src,water_price,src,NG_price,src,NG_fugitive_emissions,src,LCB_price,src,LCB_upstream_emissions,src,grid_price,src,grid_emissions_intensity,src,renew_price,src,renew_emissions_intensity,src,nuke_price,src,nuke_emissions_intensity,src,hourly_labor_rate,src = row
                H_pathway = "n/a"
                C_pathway = "n/a"
                E_pathway = E_pathways[pathway_index]
                # Check whether we are working with grid or renewable electricity
                # if renewables, fix electricity price and emissions to imposed values
                if E_pathway == "grid":
                    elect_price = grid_price
                    elect_emissions_intensity = grid_emissions_intensity
                elif E_pathway == "renewable":
                    elect_price = renew_price
                    elect_emissions_intensity = renew_emissions_intensity
                elif E_pathway == "nuke":
                    elect_price = nuke_price
                    elect_emissions_intensity = nuke_emissions_intensity
                # Calculations use LTE pathway for all, but subtract away the LTE costs/emissions
                if process == "hydrogen_liquefaction":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_hydrogen("LTE",instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod = calculate_production_costs_emissions_STP_hydrogen("LTE",instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx -= CapEx_H2prod
                    OpEx -= OpEx_H2prod
                    emissions -= emissions_H2prod
                    fuel = "liquid_hydrogen"
                    comment = "Liquefaction of STP hydrogen to cryogenic hydrogen at atmospheric pressure"
                elif process == "hydrogen_compression":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_compressed_hydrogen("LTE",instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod = calculate_production_costs_emissions_STP_hydrogen("LTE",instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx -= CapEx_H2prod
                    OpEx -= OpEx_H2prod
                    emissions -= emissions_H2prod
                    fuel = "compressed_hydrogen"
                    comment = "compression of STP hydrogen to gaseous hydrogen at 700 bar"
                elif process == "hydrogen_to_ammonia_conversion":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_ammonia("LTE",instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx_H2prod, OpEx_H2prod, emissions_H2prod = calculate_production_costs_emissions_STP_hydrogen("LTE",instal_cost,water_price,NG_price,NG_fugitive_emissions,LCB_price,LCB_upstream_emissions,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    CapEx -= CapEx_H2prod*NH3_H2_demand
                    OpEx -= OpEx_H2prod*NH3_H2_demand
                    emissions -= emissions_H2prod*NH3_H2_demand
                    fuel = "ammonia"
                    comment = "conversion of STP hydrogen to liquid cryogenic ammonia at atmospheric pressure"
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
