Import Fleet "bulk_carrier_handy"
Fleet "bulk_carrier_handy" {
    Vessels = [Vessel("bulk_carrier_handy_ice_oil"),
               Vessel("bulk_carrier_handy_ice_ammonia"),
               Vessel("bulk_carrier_handy_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_handy_ice_methanol"),
               Vessel("bulk_carrier_handy_ice_FTdiesel"),
               Vessel("bulk_carrier_handy_ice_methane")
               ]
    InitialVessels = 1
}

Import Fleet "bulk_carrier_panamax"
Fleet "bulk_carrier_panamax" {
    Vessels = [Vessel("bulk_carrier_panamax_ice_oil"),
               Vessel("bulk_carrier_panamax_ice_ammonia"),
               Vessel("bulk_carrier_panamax_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_panamax_ice_methanol"),
               Vessel("bulk_carrier_panamax_ice_FTdiesel"),
               Vessel("bulk_carrier_panamax_ice_methane")
               ]
    InitialVessels = 4
}

Import Fleet "bulk_carrier_capesize"
Fleet "bulk_carrier_capesize" {
    Vessels = [Vessel("bulk_carrier_capesize_ice_oil"),
               Vessel("bulk_carrier_capesize_ice_ammonia"),
               Vessel("bulk_carrier_capesize_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_capesize_ice_methanol"),
               Vessel("bulk_carrier_capesize_ice_FTdiesel"),
               Vessel("bulk_carrier_capesize_ice_methane")
               ]
    InitialVessels = 2
}

#Import Fleet "container_15000_teu"
#Fleet "container_15000_teu" {
#    Vessels = [Vessel("container_15000_teu_ice_oil"),
#               Vessel("container_15000_teu_ice_ammonia"),
#               Vessel("container_15000_teu_ice_liquid_hydrogen"),
#               Vessel("container_15000_teu_ice_methanol"),
#               Vessel("container_15000_teu_ice_FTdiesel"),
#               Vessel("container_15000_teu_ice_methane")
#               ]
#    InitialVessels = 0
#}

Import Fleet "container_8000_teu"
Fleet "container_8000_teu" {
    Vessels = [Vessel("container_8000_teu_ice_oil"),
               Vessel("container_8000_teu_ice_ammonia"),
               Vessel("container_8000_teu_ice_liquid_hydrogen"),
               Vessel("container_8000_teu_ice_methanol"),
               Vessel("container_8000_teu_ice_FTdiesel"),
               Vessel("container_8000_teu_ice_methane")
               ]
    InitialVessels = 41
}

Import Fleet "container_3500_teu"
Fleet "container_3500_teu" {
    Vessels = [Vessel("container_3500_teu_ice_oil"),
               Vessel("container_3500_teu_ice_ammonia"),
               Vessel("container_3500_teu_ice_liquid_hydrogen"),
               Vessel("container_3500_teu_ice_methanol"),
               Vessel("container_3500_teu_ice_FTdiesel"),
               Vessel("container_3500_teu_ice_methane")
               ]
    InitialVessels = 18
}

#Import Fleet "tanker_300k_dwt"
#Fleet "tanker_300k_dwt" {
#    Vessels = [Vessel("tanker_300k_dwt_ice_oil"),
#               Vessel("tanker_300k_dwt_ice_ammonia"),
#               Vessel("tanker_300k_dwt_ice_liquid_hydrogen"),
#               Vessel("tanker_300k_dwt_ice_methanol"),
#               Vessel("tanker_300k_dwt_ice_FTdiesel"),
#               Vessel("tanker_300k_dwt_ice_methane")
#               ]
#    InitialVessels = 0
#}

Import Fleet "tanker_100k_dwt"
Fleet "tanker_100k_dwt" {
    Vessels = [Vessel("tanker_100k_dwt_ice_oil"),
               Vessel("tanker_100k_dwt_ice_ammonia"),
               Vessel("tanker_100k_dwt_ice_liquid_hydrogen"),
               Vessel("tanker_100k_dwt_ice_methanol"),
               Vessel("tanker_100k_dwt_ice_FTdiesel"),
               Vessel("tanker_100k_dwt_ice_methane")
               ]
    InitialVessels = 9
}

Import Fleet "tanker_35k_dwt"
Fleet "tanker_35k_dwt" {
    Vessels = [Vessel("tanker_35k_dwt_ice_oil"),
               Vessel("tanker_35k_dwt_ice_ammonia"),
               Vessel("tanker_35k_dwt_ice_liquid_hydrogen"),
               Vessel("tanker_35k_dwt_ice_methanol"),
               Vessel("tanker_35k_dwt_ice_FTdiesel"),
               Vessel("tanker_35k_dwt_ice_methane")
               ]
    InitialVessels = 18
}


Fleet "*" {
    InterFuelUptake = Uptake("inter_fuel_vessel")
    IntraFuelUptake = Uptake("intra_fuel_vessel")
    FuelConversionUptake = Uptake("fuel_conversion_uptake")
}

Uptake "intra_fuel_vessel" {
    Method = DISCRETE
    Beta = 1.
    
    CoefficientOfVariation = 0.15
}

Uptake "inter_fuel_vessel" {
    Method = DISCRETE
    Beta = 1.
    
    CoefficientOfVariation = 0.15
}

Uptake "fuel_conversion_uptake" {
    Method = DISCRETE
    Beta = 1.
    
    CoefficientOfVariation = 0.15
}
