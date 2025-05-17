Import Fleet "bulk_carrier_handy"
Fleet "bulk_carrier_handy" {
    Vessels = [Vessel("bulk_carrier_handy_ice_oil"),
               Vessel("bulk_carrier_handy_ice_ammonia"),
               #Vessel("bulk_carrier_handy_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_handy_ice_methanol")#,
               #Vessel("bulk_carrier_handy_ice_FTdiesel")
               ]
    InitialVessels = 1
}

Import Fleet "bulk_carrier_panamax"
Fleet "bulk_carrier_panamax" {
    Vessels = [Vessel("bulk_carrier_panamax_ice_oil"),
               Vessel("bulk_carrier_panamax_ice_ammonia"),
               #Vessel("bulk_carrier_panamax_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_panamax_ice_methanol")#,
               #Vessel("bulk_carrier_panamax_ice_FTdiesel")
               ]
    InitialVessels = 4
}

Import Fleet "bulk_carrier_capesize"
Fleet "bulk_carrier_capesize" {
    Vessels = [Vessel("bulk_carrier_capesize_ice_oil"),
               Vessel("bulk_carrier_capesize_ice_ammonia"),
               #Vessel("bulk_carrier_capesize_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_capesize_ice_methanol")#,
               #Vessel("bulk_carrier_capesize_ice_FTdiesel")
               ]
    InitialVessels = 2
}

