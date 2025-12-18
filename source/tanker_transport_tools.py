"""
Date: 251204
Author: danikae
Purpose: Evaluate cost and emission rates for fuels of interest
"""

from common_tools import get_top_dir, get_fuel_density, ensure_directory_exists
import pandas as pd
from modify_tanks_and_cargo_capacity import get_fuel_info_dict, calculate_modified_cargo_capacities_no_sf,  calculate_modified_cargo_capacities_with_sf, get_eff_dict, get_tank_size_factors, get_route_properties

fuels = [
    "ammonia",
    "methanol",
    "liquid_hydrogen",
    "compressed_hydrogen",
    "lng",
    "FTdiesel",
    "lsfo",
    "bio_cfp"
]

top_dir = get_top_dir()

cargo_info_df = pd.read_csv(f"{top_dir}/info_files/assumed_cargo_density.csv")

def get_modified_cargo_capacity(vessel_class, fuel, sf):
    """
    Calculates modified cargo capacities for a given vessel class operating on a given fuel type and carrying a commodity with stowage factor SF. The fuel type determines how much space is taken up by tanks and the SF determines whether the cargo is mass- or volume-limited.

    Parameters
    ----------
    vessel_class : string
        Name of the vessel class to get the modified cargo capacity for
    
    fuel : string
        Fuel name
        
    sf : float
        Stowage factor (SF) for of the commodity being carried by the vessel

    Returns
    -------
    volume_capacity_fuel : float
        Volume capacity of the vessel carrying the given commodity, in m^3
    
    mass_capacity_fuel : float
        Mass capacity of the vessel carrying the given commodity, in tonnes
    """
    mass_density_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Mass density (kg/L)"
    )
    LHV_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Lower Heating Value (MJ / kg)"
    )
    boiloff_rate_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Boil-off Rate (%/day)"
    )

    sec_dict = get_fuel_info_dict(
        f"{top_dir}/info_files/fuel_info.csv", "Reliquefaction SEC (kWh/kg)"
    )

    eff_dict = get_eff_dict(fuels + ["lsfo"])
    
    tank_size_factors_dict, days_to_empty_tank_dict = get_tank_size_factors(
        fuels, LHV_dict, mass_density_dict, eff_dict, boiloff_rate_dict, sec_dict
    )
    
    capacity_dict = calculate_modified_cargo_capacities_no_sf(
        vessel_class,
        fuel,
        cargo_info_df,
        mass_density_dict,
        tank_size_factors_dict)

    volume_capacity_fuel, mass_capacity_fuel, limitation_fuel = calculate_modified_cargo_capacities_with_sf(capacity_dict, sf)
            
    return volume_capacity_fuel, mass_capacity_fuel
    
def get_annual_tonne_miles(cargo_fuel, tanker_class):
    """
    Calculate the annual tonne-miles of a given fuel carried by all tankers of the given class in the global fleet, accounting for the fraction of loaded vs. unloaded miles
    
    Parameters
    ----------
    cargo_fuel : string
        Name of the fuel being carried by the tanker as cargo (not bunker fuel)
    
    tanker_class : string
        Name of the tanker's vessel class

    Returns
    -------
    annual_tonne_miles : float
        Annual tonne-miles carried by the vessel
    """
    
    # Get the stowage factor of the fuel being carried as cargo
    cargo_fuel_density = get_fuel_density(cargo_fuel)
    cargo_fuel_sf = 1 / cargo_fuel_density
    
    route_properties = get_route_properties(tanker_class)
    annual_miles_df = pd.read_csv(f"{top_dir}/processed_results/lsfo-fossil-Miles-fleet.csv", index_col="Region")
    annual_miles = annual_miles_df.loc["Global", tanker_class + "_ice"]
    volume_capacity, mass_capacity = get_modified_cargo_capacity(tanker_class, "lsfo", cargo_fuel_sf)
    annual_tonne_miles = (annual_miles * route_properties["ConditionDistribution"][route_properties["CapacityUtilizations"] == 1][0]) * mass_capacity
    
    return annual_tonne_miles
    
def get_annual_emissions(tanker_class):
    """
    Collect the annual emissions of the given tanker class operating on lsfo
    
    Parameters
    ----------
    tanker_class : string
        Name of the tanker's vessel class

    Returns
    -------
    annual_emissions : float
        Annual emissions of the vessel, in CO2e/year
    """
    
    annual_emissions_df = pd.read_csv(f"{top_dir}/processed_results/lsfo-fossil-TotalEquivalentWTW-fleet.csv", index_col="Region")
    return annual_emissions_df.loc["Global", tanker_class + "_ice"]
    
def get_annual_cost(tanker_class):
    """
    Collect the annual costs of the given tanker class operating on lsfo
    
    Parameters
    ----------
    tanker_class : string
        Name of the tanker's vessel class

    Returns
    -------
    annual_costs : float
        Annualized lifecycle costs of the vessel, in 2024 USD/year
    """
    
    annual_cost_df = pd.read_csv(f"{top_dir}/processed_results/lsfo-fossil-TotalCost-fleet.csv", index_col="Region")
    return annual_cost_df.loc["Global", tanker_class + "_ice"]
    
def tanker_shipping_cost_per_tonne_mile(cargo_fuel):
    """
    Calculate the average cost to ship the given fuel as cargo using tankers.
    
    Parameters
    ----------
    cargo_fuel : string
        Name of the tanker's vessel class

    Returns
    -------
    cost_per_tonne_mile : float
        Cost to ship the fuel one tonne-mile, in 2024 USD / tonne-mile
    """
    
    annual_tonne_miles_all_tankers = 0
    annual_cost_all_tankers = 0
    
    for tanker_class in ["tanker_100k_dwt", "tanker_300k_dwt", "tanker_35k_dwt"]:
        annual_tonne_miles_all_tankers += get_annual_tonne_miles(cargo_fuel, tanker_class)
        annual_cost_all_tankers += get_annual_cost(tanker_class)
        
    cost_per_tonne_mile = annual_cost_all_tankers / annual_tonne_miles_all_tankers
        
    return cost_per_tonne_mile
    
def tanker_shipping_emissions_per_tonne_mile(cargo_fuel):
    """
    Calculate the average emissions to ship the given fuel as cargo using tankers.
    
    Parameters
    ----------
    cargo_fuel : string
        Name of the tanker's vessel class

    Returns
    -------
    emissions_per_tonne_mile : float
        Emissions to ship the fuel one tonne-mile, in  / tonne-mile
    """
    
    annual_tonne_miles_all_tankers = 0
    annual_emissions_all_tankers = 0
    
    for tanker_class in ["tanker_100k_dwt", "tanker_300k_dwt", "tanker_35k_dwt"]:
        annual_tonne_miles_all_tankers += get_annual_tonne_miles(cargo_fuel, tanker_class)
        annual_emissions_all_tankers += get_annual_emissions(tanker_class)
        
    emissions_per_tonne_mile = annual_emissions_all_tankers / annual_tonne_miles_all_tankers
        
    return emissions_per_tonne_mile

## Main function for testing, should be commented out in general
#def main():
#    print(tanker_shipping_cost_per_tonne_mile("liquid_hydrogen"))
#    print(tanker_shipping_emissions_per_tonne_mile("liquid_hydrogen"))
#
#if __name__ == "__main__":
#    main()
    
