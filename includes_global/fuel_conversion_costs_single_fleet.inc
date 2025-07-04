# Fuel conversion costs
Fleet "singapore_rotterdam" {
    # Handy bulk carrier
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_methanol", 14.3e6)
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_methane", 14.3e6)  # DME: copied from methanol
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_ammonia", 19.0e6)
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_liquid_hydrogen", 38.0e6)  # DME: Double ammonia
    set_fuel_conversion_cost("bulk_carrier_handy_ice_methane", "bulk_carrier_handy_ice_ammonia", 11.0e6)

    # Panamax bulk carrier
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_methanol", 17.2e6)
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_methane", 17.2e6)  # DME: copied from methanol
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_ammonia", 23.7e6)
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_liquid_hydrogen", 47.4e6)  # DME: Double ammonia
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_methane", "bulk_carrier_panamax_ice_ammonia", 14.1e6)

    # Capesize bulk carrier
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_methanol", 25.4e6)
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_methane", 25.4e6)  # DME: copied from methanol
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_ammonia", 33.1e6)
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_liquid_hydrogen", 66.2e6)  # DME: Double ammonia
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_methane", "bulk_carrier_capesize_ice_ammonia", 17.0e6)

    # 8000 TEU Container ship
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_methanol", 17.1e6)
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_methane", 17.1e6)  # DME: copied from methanol
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_ammonia", 25.6e6)
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_liquid_hydrogen", 51.2e6)  # DME: Double ammonia
    set_fuel_conversion_cost("container_8000_teu_ice_methane", "container_8000_teu_ice_ammonia", 21.5e6)

    # 3500 TEU Container ship
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_methanol", 12.4e6)
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_methane", 12.4e6)  # DME: copied from methanol
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_ammonia", 18.6e6)
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_liquid_hydrogen", 37.2e6)  # DME: Double ammonia
    set_fuel_conversion_cost("container_3500_teu_ice_methane", "container_3500_teu_ice_ammonia", 18.7e6)

    # 100k DWT tanker
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_methanol", 16.2e6)
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_methane", 16.2e6)  # DME: copied from methanol
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_ammonia", 25.1e6)
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_liquid_hydrogen", 50.2e6)  # DME: Double ammonia
    set_fuel_conversion_cost("tanker_100k_dwt_ice_methane", "tanker_100k_dwt_ice_ammonia", 18.8e6)

    # 35k DWT tanker
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_methanol", 11.4e6)
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_methane", 11.4e6)  # DME: copied from methanol
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_ammonia", 17.3e6)
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_liquid_hydrogen", 34.6e6)  # DME: Double ammonia
    set_fuel_conversion_cost("tanker_35k_dwt_ice_methane", "tanker_35k_dwt_ice_ammonia", 17.3e6)
}
