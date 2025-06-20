Fleet "bulk_carrier" {
    Vessels = [
               Vessel("bulk_carrier_handy_ice_oil"),
               Vessel("bulk_carrier_handy_ice_ammonia"),
               Vessel("bulk_carrier_handy_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_handy_ice_methanol"),
               Vessel("bulk_carrier_handy_ice_FTdiesel"),
               Vessel("bulk_carrier_handy_ice_methane"),
               Vessel("bulk_carrier_panamax_ice_oil"),
               Vessel("bulk_carrier_panamax_ice_ammonia"),
               Vessel("bulk_carrier_panamax_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_panamax_ice_methanol"),
               Vessel("bulk_carrier_panamax_ice_FTdiesel"),
               Vessel("bulk_carrier_panamax_ice_methane"),
               Vessel("bulk_carrier_capesize_ice_oil"),
               Vessel("bulk_carrier_capesize_ice_ammonia"),
               Vessel("bulk_carrier_capesize_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_capesize_ice_methanol"),
               Vessel("bulk_carrier_capesize_ice_FTdiesel"),
               Vessel("bulk_carrier_capesize_ice_methane")
               ]
    # 1 bulk_carrier_handy ; 4 bulk_carrier_panamax ; 2 bulk_carrier_capesize
    InitialVessels = 1.15
    
    InitialSplit = [0.15652173913043478261,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.55652173913043478261,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.28695652173913043478,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000
                    ]

    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.020, 0.017, 0.015, 0.018, 0.022, 0.020, 0.025, 0.030, 0.051, 0.084, 0.095, 0.092, 0.057, 0.046, 0.058, 0.047, 0.039, 0.024, 0.029, 0.030, 0.031, 0.031, 0.035, 0.046, 0.039],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.033, 0.017, 0.008, 0.025, 0.029, 0.034, 0.026, 0.024, 0.025, 0.055, 0.085, 0.105, 0.077, 0.050, 0.039, 0.037, 0.035, 0.022, 0.045, 0.047, 0.034, 0.038, 0.040, 0.034, 0.036],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.008, 0.005, 0.013, 0.020, 0.023, 0.029, 0.028, 0.022, 0.055, 0.105, 0.124, 0.107, 0.051, 0.047, 0.044, 0.051, 0.037, 0.025, 0.041, 0.056, 0.043, 0.021, 0.017, 0.012, 0.015],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    ]

    # trade growth
    TradeGrowth = 0
    
    # orderbook. DME: Setting all orderbooks empty until I can come up with actual orderbook estimates on the S-R corridor
    Orderbooks = [
                  Forecast("bulk_carrier_handy_orderbook_oil"),
                  Forecast("bulk_carrier_handy_orderbook_ammonia"),
                  Forecast("bulk_carrier_handy_orderbook_liquid_hydrogen"),
                  Forecast("bulk_carrier_handy_orderbook_methanol"),
                  Forecast("bulk_carrier_handy_orderbook_diesel"),
                  Forecast("bulk_carrier_handy_orderbook_methane"),
                  Forecast("bulk_carrier_panamax_orderbook_oil"),
                  Forecast("bulk_carrier_panamax_orderbook_ammonia"),
                  Forecast("bulk_carrier_panamax_orderbook_liquid_hydrogen"),
                  Forecast("bulk_carrier_panamax_orderbook_methanol"),
                  Forecast("bulk_carrier_panamax_orderbook_diesel"),
                  Forecast("bulk_carrier_panamax_orderbook_methane"),
                  Forecast("bulk_carrier_capesize_orderbook_oil"),
                  Forecast("bulk_carrier_capesize_orderbook_ammonia"),
                  Forecast("bulk_carrier_capesize_orderbook_liquid_hydrogen"),
                  Forecast("bulk_carrier_capesize_orderbook_methanol"),
                  Forecast("bulk_carrier_capesize_orderbook_diesel"),
                  Forecast("bulk_carrier_capesize_orderbook_methane")
                  ]
    
    ######### fuel conversions #########
    # Handy bulk carrier
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_methanol", 14.3e6)
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_methane", 14.3e6)  # DME: copied from methanol
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_ammonia", 19.0e6)
    set_fuel_conversion_cost("bulk_carrier_handy_ice_oil", "bulk_carrier_handy_ice_liquid_hydrogen", 19.0e6)  # DME: Copied from ammonia
    set_fuel_conversion_cost("bulk_carrier_handy_ice_methane", "bulk_carrier_handy_ice_ammonia", 11.0e6)

    # Panamax bulk carrier
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_methanol", 17.2e6)
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_methane", 17.2e6)  # DME: copied from methanol
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_ammonia", 23.7e6)
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_oil", "bulk_carrier_panamax_ice_liquid_hydrogen", 23.7e6)  # DME: Copied from ammonia
    set_fuel_conversion_cost("bulk_carrier_panamax_ice_methane", "bulk_carrier_panamax_ice_ammonia", 14.1e6)

    # Capesize bulk carrier
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_methanol", 25.4e6)
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_methane", 25.4e6)  # DME: copied from methanol
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_ammonia", 33.1e6)
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_oil", "bulk_carrier_capesize_ice_liquid_hydrogen", 33.1e6)  # DME: Copied from ammonia
    set_fuel_conversion_cost("bulk_carrier_capesize_ice_methane", "bulk_carrier_capesize_ice_ammonia", 17.0e6)
    ####################################
}

Fleet "container" {
    Vessels = [
               Vessel("container_8000_teu_ice_oil"),
               Vessel("container_8000_teu_ice_ammonia"),
               Vessel("container_8000_teu_ice_liquid_hydrogen"),
               Vessel("container_8000_teu_ice_methanol"),
               Vessel("container_8000_teu_ice_FTdiesel"),
               Vessel("container_8000_teu_ice_methane"),
               Vessel("container_3500_teu_ice_oil"),
               Vessel("container_3500_teu_ice_ammonia"),
               Vessel("container_3500_teu_ice_liquid_hydrogen"),
               Vessel("container_3500_teu_ice_methanol"),
               Vessel("container_3500_teu_ice_FTdiesel"),
               Vessel("container_3500_teu_ice_methane")
               ]
    # 41 container_8000_teu ; 18 container_3500_teu
    InitialVessels = 7.14

    InitialSplit = [0.67366946778711484594,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.32633053221288515406,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000
                    ]

    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]

    InitialAgeSplit = Array [
        [0.018, 0.025, 0.028, 0.026, 0.044, 0.083, 0.064, 0.080, 0.053, 0.064, 0.060, 0.045, 0.066, 0.065, 0.080, 0.035, 0.029, 0.018, 0.002, 0.001, 0.016, 0.016, 0.022, 0.049, 0.010],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.033, 0.037, 0.032, 0.037, 0.056, 0.070, 0.086, 0.090, 0.058, 0.045, 0.028, 0.033, 0.031, 0.026, 0.023, 0.018, 0.022, 0.026, 0.028, 0.026, 0.027, 0.035, 0.057, 0.062, 0.013],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    ]

    # trade growth
    TradeGrowth = 0

    # orderbook. DME: Setting all orderbooks empty until I can come up with actual orderbook estimates on the S-R corridor
    Orderbooks = [
                  Forecast("container_8000_teu_orderbook_oil"),
                  Forecast("container_8000_teu_orderbook_ammonia"),
                  Forecast("container_8000_teu_orderbook_liquid_hydrogen"),
                  Forecast("container_8000_teu_orderbook_methanol"),
                  Forecast("container_8000_teu_orderbook_diesel"),
                  Forecast("container_8000_teu_orderbook_methane"),
                  Forecast("container_3500_teu_orderbook_oil"),
                  Forecast("container_3500_teu_orderbook_ammonia"),
                  Forecast("container_3500_teu_orderbook_liquid_hydrogen"),
                  Forecast("container_3500_teu_orderbook_methanol"),
                  Forecast("container_3500_teu_orderbook_diesel"),
                  Forecast("container_3500_teu_orderbook_methane")
                  ]

    ######### fuel conversions #########
    # 8000 TEU Container ship
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_methanol", 17.1e6)
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_methane", 17.1e6)  # DME: copied from methanol
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_ammonia", 25.6e6)
    set_fuel_conversion_cost("container_8000_teu_ice_oil", "container_8000_teu_ice_liquid_hydrogen", 25.6e6)  # DME: Copied from ammonia
    set_fuel_conversion_cost("container_8000_teu_ice_methane", "container_8000_teu_ice_ammonia", 21.5e6)

    # 3500 TEU Container ship
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_methanol", 12.4e6)
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_methane", 12.4e6)  # DME: copied from methanol
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_ammonia", 18.6e6)
    set_fuel_conversion_cost("container_3500_teu_ice_oil", "container_3500_teu_ice_liquid_hydrogen", 18.6e6)  # DME: Copied from ammonia
    set_fuel_conversion_cost("container_3500_teu_ice_methane", "container_3500_teu_ice_ammonia", 18.7e6)
    ####################################
}

Fleet "tanker" {
    Vessels = [
               Vessel("tanker_100k_dwt_ice_oil"),
               Vessel("tanker_100k_dwt_ice_ammonia"),
               Vessel("tanker_100k_dwt_ice_liquid_hydrogen"),
               Vessel("tanker_100k_dwt_ice_methanol"),
               Vessel("tanker_100k_dwt_ice_FTdiesel"),
               Vessel("tanker_100k_dwt_ice_methane"),
               Vessel("tanker_35k_dwt_ice_oil"),
               Vessel("tanker_35k_dwt_ice_ammonia"),
               Vessel("tanker_35k_dwt_ice_liquid_hydrogen"),
               Vessel("tanker_35k_dwt_ice_methanol"),
               Vessel("tanker_35k_dwt_ice_FTdiesel"),
               Vessel("tanker_35k_dwt_ice_methane")
               ]
    # 9 tanker_100k_dwt ; 18 tanker_35k_dwt
    InitialVessels = 4.31

    InitialSplit = [0.32250580046403712297,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.67749419953596287703,
                    0.000,
                    0.000,
                    0.000,
                    0.000,
                    0.000
                    ]

    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]

    InitialAgeSplit = Array [
        [0.008, 0.022, 0.054, 0.054, 0.068, 0.055, 0.062, 0.070, 0.084, 0.063, 0.056, 0.038, 0.020, 0.013, 0.022, 0.045, 0.053, 0.038, 0.038, 0.018, 0.028, 0.021, 0.015, 0.016, 0.039],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.016, 0.014, 0.017, 0.017, 0.017, 0.016, 0.020, 0.030, 0.033, 0.036, 0.045, 0.053, 0.067, 0.063, 0.050, 0.046, 0.051, 0.044, 0.038, 0.035, 0.034, 0.033, 0.037, 0.033, 0.030, 0.031, 0.029, 0.021, 0.026, 0.019],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    ]

    # trade growth
    TradeGrowth = 0

    # orderbook. DME: Setting all orderbooks empty until I can come up with actual orderbook estimates on the S-R corridor
    Orderbooks = [
                  Forecast("tanker_100k_dwt_orderbook_oil"),
                  Forecast("tanker_100k_dwt_orderbook_ammonia"),
                  Forecast("tanker_100k_dwt_orderbook_liquid_hydrogen"),
                  Forecast("tanker_100k_dwt_orderbook_methanol"),
                  Forecast("tanker_100k_dwt_orderbook_diesel"),
                  Forecast("tanker_100k_dwt_orderbook_methane"),
                  Forecast("tanker_35k_dwt_orderbook_oil"),
                  Forecast("tanker_35k_dwt_orderbook_ammonia"),
                  Forecast("tanker_35k_dwt_orderbook_liquid_hydrogen"),
                  Forecast("tanker_35k_dwt_orderbook_methanol"),
                  Forecast("tanker_35k_dwt_orderbook_diesel"),
                  Forecast("tanker_35k_dwt_orderbook_methane")
                  ]

    ######### fuel conversions #########
    # 100k DWT tanker
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_methanol", 16.2e6)
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_methane", 16.2e6)  # DME: copied from methanol
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_ammonia", 25.1e6)
    set_fuel_conversion_cost("tanker_100k_dwt_ice_oil", "tanker_100k_dwt_ice_liquid_hydrogen", 25.1e6)  # DME: Copied from ammonia
    set_fuel_conversion_cost("tanker_100k_dwt_ice_methane", "tanker_100k_dwt_ice_ammonia", 18.8e6)

    # 35k DWT tanker
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_methanol", 11.4e6)
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_methane", 11.4e6)  # DME: copied from methanol
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_ammonia", 17.3e6)
    set_fuel_conversion_cost("tanker_35k_dwt_ice_oil", "tanker_35k_dwt_ice_liquid_hydrogen", 17.3e6)  # DME: Copied from ammonia
    set_fuel_conversion_cost("tanker_35k_dwt_ice_methane", "tanker_35k_dwt_ice_ammonia", 17.3e6)
    ####################################
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

######### orderbook forecasts #########

## Handy bulk carrier ##
Forecast "bulk_carrier_handy_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_handy_orderbook_ammonia" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_handy_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_handy_orderbook_methanol" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_handy_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_handy_orderbook_methane" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## Panamax bulk carrier ##
Forecast "bulk_carrier_panamax_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_panamax_orderbook_ammonia" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_panamax_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_panamax_orderbook_methanol" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_panamax_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_panamax_orderbook_methane" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## Capesize bulk carrier ##
Forecast "bulk_carrier_capesize_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_capesize_orderbook_ammonia" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_capesize_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_capesize_orderbook_methanol" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_capesize_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "bulk_carrier_capesize_orderbook_methane" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## 8000 TEU container ##
Forecast "container_8000_teu_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_8000_teu_orderbook_ammonia" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_8000_teu_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_8000_teu_orderbook_methanol" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_8000_teu_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_8000_teu_orderbook_methane" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## 3500 TEU container ##
Forecast "container_3500_teu_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_3500_teu_orderbook_ammonia" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_3500_teu_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_3500_teu_orderbook_methanol" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_3500_teu_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "container_3500_teu_orderbook_methane" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## 100k DWT tanker ##
Forecast "tanker_100k_dwt_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_100k_dwt_orderbook_ammonia" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_100k_dwt_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_100k_dwt_orderbook_methanol" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_100k_dwt_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_100k_dwt_orderbook_methane" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## 35k DWT tanker ##
Forecast "tanker_35k_dwt_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_35k_dwt_orderbook_ammonia" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_35k_dwt_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_35k_dwt_orderbook_methanol" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_35k_dwt_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
Forecast "tanker_35k_dwt_orderbook_methane" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
####################################
