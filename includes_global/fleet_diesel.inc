# Definition of the diesel bulk fleet
Fleet "diesel_fleet_bulk" {
    Vessels = [Vessel("bulk_carrier_capesize_ice_diesel"), Vessel("bulk_carrier_handy_ice_diesel"), Vessel("bulk_carrier_panamax_ice_diesel")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 12583      
    InitialSplit = [0.16, 0.33, 0.51] # Vessel numbers taken from the default files bulk_carrier_capesize.inc, bulk_carrier_handy.inc and bulk_carrier_panamax.inc
    TradeGrowth = 0
}

# Definition of the diesel container fleet
Fleet "diesel_fleet_container" {
    Vessels = [Vessel("container_15000_teu_ice_diesel"), Vessel("container_3500_teu_ice_diesel"), Vessel("container_8000_teu_ice_diesel")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 5565      
    InitialSplit = [0.08, 0.70, 0.22] # Vessel numbers taken from the default files container_15000_TEU.inc, container_3500_TEU.inc and container_8000_TEU.inc
    TradeGrowth = 0
}

# Definition of the diesel container fleet
Fleet "diesel_fleet_tanker" {
    Vessels = [Vessel("tanker_100k_dwt_ice_diesel"), Vessel("tanker_300k_dwt_ice_diesel"), Vessel("tanker_35k_dwt_ice_diesel")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 5565      
    InitialSplit = [0.28, 0.07, 0.65] # Vessel numbers taken from the default files tanker_100k_dwt.inc, tanker_300k_dwt.inc and tanker_35k_dwt.inc
    TradeGrowth = 0
}

# Definition of the diesel gas carrier fleet
Fleet "diesel_fleet_gas" {
    Vessels = [Vessel("gas_carrier_100k_cbm_ice_diesel")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 2156
    InitialSplit = [1] # Vessel numbers taken from the default file gas_carrier_100k_cbm_ice_oil.inc
    TradeGrowth = 0
}

# Define uptake node
Uptake "uptake" {
    Objective=MINIMIZE
}

# Need to enforce alternative fuel use in all vessels
Import Vessel "bulk_carrier_capesize_ice_diesel"
Vessel "bulk_carrier_capesize_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "bulk_carrier_handy_ice_diesel"
Vessel "bulk_carrier_handy_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "bulk_carrier_panamax_ice_diesel"
Vessel "bulk_carrier_panamax_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "container_15000_teu_ice_diesel"
Vessel "container_15000_teu_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "container_3500_teu_ice_diesel"
Vessel "container_3500_teu_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "container_8000_teu_ice_diesel"
Vessel "container_8000_teu_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "tanker_100k_dwt_ice_diesel"
Vessel "tanker_100k_dwt_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "tanker_300k_dwt_ice_diesel"
Vessel "tanker_300k_dwt_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "tanker_35k_dwt_ice_diesel"
Vessel "tanker_35k_dwt_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
Import Vessel "gas_carrier_100k_cbm_ice_diesel"
Vessel "gas_carrier_100k_cbm_ice_diesel" {
    EnforceMainFuelShare = TRUE
}
