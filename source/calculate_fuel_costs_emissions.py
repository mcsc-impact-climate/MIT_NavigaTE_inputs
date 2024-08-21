"""
Date: Aug 21, 2024
Purpose: Prepare .csv files contained in input_fuel_pathway_data using consistent assumptions.
"""

import pandas as pd
import os
from common_tools import get_top_dir

# Function to calculate CapEx, OpEx, LCOF, and production GHG emissions for STP hydrogen
workhours_per_year = 52*40 # number of work-hours per year
NG_HHV = 0.0522 # GJ/kg NG
NG_GWP = 28 # GWP100 of methane (surrogate for NG) using IPCC-AR5 as in MEPC.391(81)
gen_admin_rate = 0.2 # 20% G&A rate
op_maint_rate = 0.04 # O&M rate
tax_rate = 0.02 # 2% tax rate
NG_percent_fugitive = 0.02 # 2% fugitive NG emissions
renew_price = 0.04 # [2024$/kWh] price of renewable electricity
CO2_price = 0.04 # [2024$/kg CO2] price of biogenic CO2 (e.g. ~$40/tonne from biomass BEC, ~$90/tonne from biogas BEC, ~$200+/tonne from DAC)
CO2_upstream_emissions = 0.02 # [kg CO2e/kg CO2] upstream emissions from CO2 capture (e.g. 0.02 from biomass BEC, 0.05 from biogas BEC..)
renew_emissions_intensity = 0.01 # [kg CO2e/kWh] upstream emissions from renewable electricity

# Inputs for STP H2 production from low-temperature water electrolysis
H2_LTE_elect_demand = 51.03004082 # [kWh elect/kg H2] from Aspen Plus
H2_LTE_NG_demand = 0 # [GJ NG/kg H2] zero for LTE
H2_LTE_water_demand = 0.01430886 # [m^3 H2O/kg H2] from Aspen Plus
H2_LTE_base_CapEx = 0.56079 # [2024$/kg H2] from H2A
H2_LTE_full_time_employed = 10 # [full-time employees] from H2A
H2_LTE_yearly_output = 20003825 # [kg H2/year] from Aspen Plus
H2_LTE_onsite_emissions = 0 # [kg CO2e/kg H2] zero for LTE

# Inputs for STP H2 production from steam methane reforming with 96% CO2 capture rate
H2_SMRCCS_elect_demand = 0.041/20125 # [kWh elect/kg H2] from Zang et al 2021
H2_SMRCCS_NG_demand = 3947/20125 # [GJ NG/kg H2] from Zang et al 2021
H2_SMRCCS_water_demand = (308.347 + 177.085)/20125 # [m^3 H2O/kg H2] from Zang et al 2021
H2_SMRCCS_base_CapEx = 1.2253*1419/365*0.1018522 # [2024$/kg] amortized TPC from Zang et al 2021
H2_SMRCCS_full_time_employed = 22 # [full-time employees] from H2A
H2_SMRCCS_yearly_output = 365*24*20125 # [kg H2/year] from Zang et al 2021
H2_SMRCCS_onsite_emissions = 7656/20125 # [kg CO2e output/kg H2] from Zang et al 2021

# Inputs for STP H2 production from steam methane reforming without CO2 capture
H2_SMR_elect_demand = 0.013/20126 # [kWh elect/kg H2] from Zang et al 2021
H2_SMR_NG_demand = (3712 - 514/0.8)/20126 # [GJ NG/kg H2] (including steam displacement) from Zang et al 2021
H2_SMR_water_demand = 336.728/20126 # [m^3 H2O/kg H2] from Zang et al 2021
H2_SMR_base_CapEx = 1.2253*564/365*0.1018522 # [2024$/kg] amortized TPC from Zang et al 2021
H2_SMR_full_time_employed = 18 # [full-time employees] from H2A
H2_SMR_yearly_output = 365*24*20126 # [kg H2/year] from Zang et al 2021
H2_SMR_onsite_emissions = 188221/20126 # [kg CO2e/kgH2] from Zang et al 2021

# Inputs for production of liquid H2 at 20 K
H2_liq_base_CapEx = 0.59898 # [2024$/kg]
H2_liq_elect_demand = 8.0 # [kWh elect/kg H2]

# Inputs for production of gaseous H2 at 700 bar
H2_comp_base_CapEx = 0.17114 # [2024$/kg]
H2_comp_elect_demand = 3.0 # [kWh elect/kg H2]

# Inputs for liquid NH3 production from arbitrary H2 feedstock
NH3_elect_demand = 10.05314189 # [kWh elect/kg NH3] for LTE ammonia process from Aspen Plus
NH3_H2_demand = 2.01568/17.03022 # [kg H2/kg NH3] for LTE ammonia process from Aspen Plus
NH3_elect_demand -= 3/2*H2_LTE_elect_demand*NH3_H2_demand # subtract electrical demand from LTE H2 process
NH3_NG_demand = 0 # [GJ NG/kg H2] from Aspen Plus
NH3_water_demand = 0.00261861625758975 # [m^3 H2O/kg NH3] for LTE ammonia process from Aspen Plus
NH3_water_demand -= 3/2*H2_LTE_water_demand*NH3_H2_demand # subtract water demand from LTE H2 process
NH3_base_CapEx = 0.193395 # [2024$/kg] from H2A
NH3_base_CapEx -= 3/2*H2_LTE_base_CapEx*NH3_H2_demand # subtract base CapEx from LTE H2 process
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

def calculate_production_costs_emissions_STP_hydrogen(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate):
    if fuel_pathway.startswith("LTE_"):
        elect_demand = H2_LTE_elect_demand
        NG_demand = H2_LTE_NG_demand
        water_demand = H2_LTE_water_demand
        base_CapEx = H2_LTE_base_CapEx
        full_time_employed = H2_LTE_full_time_employed
        yearly_output = H2_LTE_yearly_output
        onsite_emissions = H2_LTE_onsite_emissions
    elif fuel_pathway.startswith("SMRCCS_"):
        elect_demand = H2_SMRCCS_elect_demand
        NG_demand = H2_SMRCCS_NG_demand
        water_demand = H2_SMRCCS_water_demand
        base_CapEx = H2_SMRCCS_base_CapEx
        full_time_employed = H2_SMRCCS_full_time_employed
        yearly_output = H2_SMRCCS_yearly_output
        onsite_emissions = H2_SMRCCS_onsite_emissions
    elif fuel_pathway.startswith("SMR_"):
        elect_demand = H2_SMR_elect_demand
        NG_demand = H2_SMR_NG_demand
        water_demand = H2_SMR_water_demand
        base_CapEx = H2_SMR_base_CapEx
        full_time_employed = H2_SMR_full_time_employed
        yearly_output = H2_SMR_yearly_output
        onsite_emissions = H2_SMR_onsite_emissions
    # calculate production values
    CapEx = base_CapEx*instal_factor
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) \
                + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity \
                + NG_demand/NG_HHV*NG_GWP*NG_percent_fugitive \
                + onsite_emissions
    
    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_liquid_hydrogen(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate):
    base_CapEx = H2_liq_base_CapEx
    elect_demand = H2_liq_elect_demand
    # calculate liquefaction values
    CapEx = base_CapEx*instal_factor
    OpEx = (op_maint_rate + tax_rate)*CapEx + elect_demand*elect_price
    emissions = elect_demand*elect_emissions_intensity
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx
    OpEx += H2_OpEx
    emissions += H2_emissions

    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_compressed_hydrogen(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate):
    base_CapEx = H2_comp_base_CapEx
    elect_demand = H2_comp_elect_demand
    # calculate compression values
    CapEx = base_CapEx*instal_factor
    OpEx = (op_maint_rate + tax_rate)*CapEx + elect_demand*elect_price
    emissions = elect_demand*elect_emissions_intensity
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += H2_CapEx
    OpEx += H2_OpEx
    emissions += H2_emissions

    return CapEx, OpEx, emissions

def calculate_production_costs_emissions_ammonia(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate):
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
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) \
                + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity \
                + NG_demand/NG_HHV*NG_GWP*NG_percent_fugitive \
                + onsite_emissions
    # add H2 feedstock cost and emissions
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += (H2_CapEx*H2_demand)
    OpEx += (H2_OpEx*H2_demand)
    emissions += (H2_emissions*H2_demand)

    return CapEx, OpEx, emissions


def calculate_production_costs_emissions_methanol(fuel_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate):
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
    Fixed_OpEx = workhours_per_year*hourly_labor_rate*full_time_employed/yearly_output*(1.0 + gen_admin_rate) \
                + (op_maint_rate + tax_rate)*CapEx
    Electricity_OpEx = elect_demand*elect_price
    NG_OpEx = NG_demand*NG_price
    Water_OpEx = water_demand*water_price
    OpEx = Fixed_OpEx + Electricity_OpEx + NG_OpEx + Water_OpEx
    emissions = elect_demand*elect_emissions_intensity \
                + NG_demand/NG_HHV*NG_GWP*NG_percent_fugitive \
                + onsite_emissions
    # add H2 feedstock costs and emissions
    H2_pathway = fuel_pathway # for now... this lets us modify later (e.g. H2_pathway = fuel_pathway.partition("_H")[0])
    H2_CapEx, H2_OpEx, H2_emissions = calculate_production_costs_emissions_STP_hydrogen(H2_pathway,instal_factor,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
    CapEx += (H2_CapEx*H2_demand)
    OpEx += (H2_OpEx*H2_demand)
    emissions += (H2_emissions*H2_demand)
    # add CO2 feedstock costs and emissions
    CO2_pathway = fuel_pathway # for now... this lets us modify later (e.g. CO2_pathway = fuel_pathway.partition("_H")[2].partition("_C")[0])
    if CO2_pathway.startswith("LTE_"):
        CO2_CapEx = 0 # No CapEx because we assume biogenic CO2 is purchased externally at a fixed price 
        CO2_OpEx = CO2_price
        CO2_emissions = CO2_upstream_emissions*CO2_demand - (44.01/32.04) - onsite_emissions # biogenic credit
    elif CO2_pathway.startswith("SMRCCS_"):
        CO2_CapEx = 0 # CO2 is "free" after already paying for upstream CCS
        CO2_OpEx = 0 # CO2 is "free" after already paying for upstream CCS
        CO2_emissions = 0 # we are working with already-captured fossil CO2
    elif CO2_pathway.startswith("SMR_"):
        CO2_CapEx = 0 # assumes integrated plant with syngas conversion
        CO2_OpEx = 0 # assumes integrated plant with syngas conversion
        CO2_emissions = -(44.01/32.04) # fossil CO2 that would have been emitted by SMR is instead embodied in fuel 
    CapEx += (CO2_CapEx*CO2_demand)
    OpEx += (CO2_OpEx*CO2_demand)
    emissions += (CO2_emissions) 

    return CapEx, OpEx, emissions

def main():
    top_dir = get_top_dir()
    input_dir = f"{top_dir}/input_fuel_pathway_data/"
    output_dir = f"{top_dir}/input_fuel_pathway_data/"

    # Read the input CSV file
    input_file = 'fuel_production_inputs.csv'
    csv_file = os.path.join(input_dir, input_file)
    input_df = pd.read_csv(csv_file)

    fuels = ["liquid_hydrogen", "compressed_hydrogen", "ammonia", "methanol"]
    fuel_pathways = ["LTE_grid", "SMRCCS_grid", "SMR_grid", "LTE_renew", "SMRCCS_renew", "SMR_renew"]
    for fuel in fuels: 
        # List to hold all rows for the output CSV
        output_data = []

        # Iterate through each row in the input data and perform calculations
        for fuel_pathway in fuel_pathways:
            for index, row in input_df.iterrows():
                region,instal_cost,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate = row
                # Check whether we are working with grid or renewable electricity
                # if renewables, fix electricity price and emissions to imposed values
                if fuel_pathway.endswith("_renew"):
                    elect_price = renew_price
                    elect_emissions_intensity = renew_emissions_intensity
                if fuel == "liquid_hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_liquid_hydrogen(fuel_pathway,instal_cost,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid cryogenic hydrogen at atmospheric pressure"
                elif fuel == "compressed_hydrogen":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_compressed_hydrogen(fuel_pathway,instal_cost,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "compressed gaseous hydrogen at 700 bar"
                elif fuel == "ammonia":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_ammonia(fuel_pathway,instal_cost,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid cryogenic ammonia at atmospheric pressure"
                elif fuel == "methanol":
                    CapEx, OpEx, emissions = calculate_production_costs_emissions_methanol(fuel_pathway,instal_cost,water_price,NG_price,elect_price,elect_emissions_intensity,hourly_labor_rate)
                    comment = "Liquid methanol at STP"
                CapEx *= 1000 # convert to $/tonne
                OpEx *= 1000 # convert to $/tonne
                LCOF = CapEx + OpEx # in $/tonne
                calculated_row = [fuel, fuel_pathway, "", region, 1, 2024, CapEx, OpEx, LCOF, emissions, "", "", "", comment]
                output_data.append(calculated_row)

        # Define the output CSV column names
        output_columns = [
            "Fuel", "Pathway Name", "Pathway Description", "Region", "Number", "Year",
            "CapEx [$/tonne]", "OpEx [$/tonne]", "LCOF [$/tonne]", "Emissions [kg CO2e / kg fuel]", "Details",
            "Model(s) or publications", "Results Produced by", "Comment"
        ]

        # Create a DataFrame for the output data
        output_df = pd.DataFrame(output_data, columns=output_columns)

        # Write the output data to a CSV file
        output_file = f"{fuel}_costs_emissions.csv"
        output_df.to_csv(os.path.join(output_dir, output_file), index=False)

        print(f"Output CSV file created: {os.path.join(output_dir, output_file)}")

if __name__ == "__main__":
    main()
