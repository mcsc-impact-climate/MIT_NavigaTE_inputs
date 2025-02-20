# Definition of the bulk fleet
Fleet "fleet_bulk" {
    Vessels = [Vessel("bulk_carrier_capesize_ice_oil"), Vessel("bulk_carrier_handy_ice_oil"), Vessel("bulk_carrier_panamax_ice_oil"),
               Vessel("bulk_carrier_capesize_ice_ammonia"), Vessel("bulk_carrier_handy_ice_ammonia"), Vessel("bulk_carrier_panamax_ice_ammonia"),
               Vessel("bulk_carrier_capesize_ice_compressed_hydrogen"), Vessel("bulk_carrier_handy_ice_compressed_hydrogen"), Vessel("bulk_carrier_panamax_ice_compressed_hydrogen"),
               Vessel("bulk_carrier_capesize_ice_liquid_hydrogen"), Vessel("bulk_carrier_handy_ice_liquid_hydrogen"), Vessel("bulk_carrier_panamax_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_capesize_ice_methanol"), Vessel("bulk_carrier_handy_ice_methanol"), Vessel("bulk_carrier_panamax_ice_methanol"),
               Vessel("bulk_carrier_capesize_ice_FTdiesel"), Vessel("bulk_carrier_handy_ice_FTdiesel"), Vessel("bulk_carrier_panamax_ice_FTdiesel"),
              ]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 12583      
    InitialSplit = [0.16, 0.33, 0.51, 0, 0, 0] # Vessel numbers taken from the default files bulk_carrier_capesize.inc, bulk_carrier_handy.inc and bulk_carrier_panamax.inc
    TradeGrowth = 0
}

# Definition of the container fleet
Fleet "fleet_container" {
    Vessels = [Vessel("container_15000_teu_ice_oil"), Vessel("container_3500_teu_ice_oil"), Vessel("container_8000_teu_ice_oil"),
               Vessel("container_15000_teu_ice_ammonia"), Vessel("container_3500_teu_ice_ammonia"), Vessel("container_8000_teu_ice_ammonia"),
               Vessel("container_15000_teu_ice_compressed_hydrogen"), Vessel("container_3500_teu_ice_compressed_hydrogen"), Vessel("container_8000_teu_ice_compressed_hydrogen"),
               Vessel("container_15000_teu_ice_liquid_hydrogen"), Vessel("container_3500_teu_ice_liquid_hydrogen"), Vessel("container_8000_teu_ice_liquid_hydrogen"),
               Vessel("container_15000_teu_ice_methanol"), Vessel("container_3500_teu_ice_methanol"), Vessel("container_8000_teu_ice_methanol"),
               Vessel("container_15000_teu_ice_FTdiesel"), Vessel("container_3500_teu_ice_FTdiesel"), Vessel("container_8000_teu_ice_FTdiesel"),
              ]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 5565      
    InitialSplit = [0.08, 0.70, 0.22, 0, 0, 0] # Vessel numbers taken from the default files container_15000_TEU.inc, container_3500_TEU.inc and container_8000_TEU.inc
    TradeGrowth = 0
}

# Definition of the tanker fleet
Fleet "fleet_tanker" {
    Vessels = [Vessel("tanker_100k_dwt_ice_oil"), Vessel("tanker_300k_dwt_ice_oil"), Vessel("tanker_35k_dwt_ice_oil"),
               Vessel("tanker_100k_dwt_ice_ammonia"), Vessel("tanker_300k_dwt_ice_ammonia"), Vessel("tanker_35k_dwt_ice_ammonia"),
               Vessel("tanker_100k_dwt_ice_compressed_hydrogen"), Vessel("tanker_300k_dwt_ice_compressed_hydrogen"), Vessel("tanker_35k_dwt_ice_compressed_hydrogen"),
               Vessel("tanker_100k_dwt_ice_liquid_hydrogen"), Vessel("tanker_300k_dwt_ice_liquid_hydrogen"), Vessel("tanker_35k_dwt_ice_liquid_hydrogen"),
               Vessel("tanker_100k_dwt_ice_methanol"), Vessel("tanker_300k_dwt_ice_methanol"), Vessel("tanker_35k_dwt_ice_methanol"),
               Vessel("tanker_100k_dwt_ice_FTdiesel"), Vessel("tanker_300k_dwt_ice_FTdiesel"), Vessel("tanker_35k_dwt_ice_FTdiesel"),
              ]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 5565      
    InitialSplit = [0.28, 0.07, 0.65, 0, 0, 0] # Vessel numbers taken from the default files tanker_100k_dwt.inc, tanker_300k_dwt.inc and tanker_35k_dwt.inc
    TradeGrowth = 0
}

# Definition of the ammonia gas carrier fleet
Fleet "fleet_gas" {
    Vessels = [Vessel("gas_carrier_100k_cbm_ice_oil"),
               Vessel("gas_carrier_100k_cbm_ice_ammonia"),
               Vessel("gas_carrier_100k_cbm_ice_compressed_hydrogen"),
               Vessel("gas_carrier_100k_cbm_ice_liquid_hydrogen"),
               Vessel("gas_carrier_100k_cbm_ice_methanol"),
               Vessel("gas_carrier_100k_cbm_ice_FTdiesel"),
              ]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 2156
    InitialSplit = [1, 0] # Vessel numbers taken from the default file gas_carrier_100k_cbm_ice_oil.inc
    TradeGrowth = 0
}

# Define uptake node
Uptake "uptake" {
    Objective=MINIMIZE
}
