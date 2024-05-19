Copy Vessel "bulk_carrier_panamax_ice_ammonia" "vessel"
Copy Vessel "bulk_carrier_panamax_ice_ammonia" "vessel_baseline"
Copy Vessel "bulk_carrier_panamax_ice_ammonia" "vessel_alternative"

Vessel "vessel" {
    EnforceMainFuelShare = TRUE
}

Vessel "vessel_baseline" {
}

Vessel "vessel_alternative" {
    EnforceMainFuelShare = TRUE
}
