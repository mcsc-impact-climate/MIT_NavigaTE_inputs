############################# Bulk carriers ###################################

# LSFO
Import Vessel "bulk_carrier_capesize_ice_oil"
Vessel "bulk_carrier_capesize_ice_oil" {
        Route=Route("singapore_rotterdam_bulk_carrier_capesize")
        NominalCapacity = 140077.8522122301
}

Import Vessel "bulk_carrier_panamax_ice_oil"
Vessel "bulk_carrier_panamax_ice_oil" {
        Route=Route("singapore_rotterdam_bulk_carrier_panamax")
        NominalCapacity = 62001.708506750336
}

Import Vessel "bulk_carrier_handy_ice_oil"
Vessel "bulk_carrier_handy_ice_oil" {
        Route=Route("singapore_rotterdam_bulk_carrier_handy")
        NominalCapacity = 21573.978761473172
}

############################# Container ships ###################################
 
# LSFO
Import Vessel "container_8000_teu_ice_oil"
Vessel "container_8000_teu_ice_oil" {
        Route=Route("singapore_rotterdam_container_8000_teu")
        NominalCapacity = 8000.0
}

Import Vessel "container_3500_teu_ice_oil"
Vessel "container_3500_teu_ice_oil" {
        Route=Route("singapore_rotterdam_container_3500_teu")
        NominalCapacity = 3500.0
}
############################# Tankers ###################################

# LSFO
Import Vessel "tanker_100k_dwt_ice_oil"
Vessel "tanker_100k_dwt_ice_oil" {
        Route=Route("singapore_rotterdam_tanker_100k_dwt")
        NominalCapacity = 95241.20647996894
}

Import Vessel "tanker_35k_dwt_ice_oil"
Vessel "tanker_35k_dwt_ice_oil" {
        Route=Route("singapore_rotterdam_tanker_35k_dwt")
        NominalCapacity = 33334.422267989125
}

