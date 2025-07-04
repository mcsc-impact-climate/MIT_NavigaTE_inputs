Fleet "bulk_carrier_capesize" {
    # vessel types
    Vessels = [Vessel("bulk_carrier_capesize_ice_oil")]
    
    # initial conditions
    InitialVessels = 0.33
    
    InitialSplit = [1.000
                    ]
    
    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.008, 0.005, 0.013, 0.020, 0.023, 0.029, 0.028, 0.022, 0.055, 0.105, 0.124, 0.107, 0.051, 0.047, 0.044, 0.051, 0.037, 0.025, 0.041, 0.056, 0.043, 0.021, 0.017, 0.012, 0.015]
    ]
    
    # trade growth
    TradeGrowth = 0 #Forecast("bulk_carrier_capesize_trade_growth")
    
    # orderbook
    Orderbooks = [Forecast("bulk_carrier_capesize_orderbook_oil")
                  ]
}

Forecast "bulk_carrier_capesize_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
        "01-01-2028"    0
    ]
    
    Extrapolate = FLAT
}

Fleet "bulk_carrier_handy" {
    # vessel types
    Vessels = [Vessel("bulk_carrier_handy_ice_oil")]
    
    # initial conditions
    InitialVessels = 0.18
    
    InitialSplit = [1.000
                    ]
    
    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.020, 0.017, 0.015, 0.018, 0.022, 0.020, 0.025, 0.030, 0.051, 0.084, 0.095, 0.092, 0.057, 0.046, 0.058, 0.047, 0.039, 0.024, 0.029, 0.030, 0.031, 0.031, 0.035, 0.046, 0.039]
    ]
    
    # trade growth
    TradeGrowth = 0 #Forecast("bulk_carrier_handy_trade_growth")
    
    # orderbook
    Orderbooks = [Forecast("bulk_carrier_handy_orderbook_oil")
                  ]
}

Forecast "bulk_carrier_handy_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

Fleet "bulk_carrier_panamax" {
    # vessel types
    Vessels = [Vessel("bulk_carrier_panamax_ice_oil")]
    
    # initial conditions
    InitialVessels = 0.64
    
    InitialSplit = [1.000
                    ]
    
    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.033, 0.017, 0.008, 0.025, 0.029, 0.034, 0.026, 0.024, 0.025, 0.055, 0.085, 0.105, 0.077, 0.050, 0.039, 0.037, 0.035, 0.022, 0.045, 0.047, 0.034, 0.038, 0.040, 0.034, 0.036]
    ]

    # trade growth
    TradeGrowth = 0 #Forecast("bulk_carrier_panamax_trade_growth")

    # orderbook
    Orderbooks = [Forecast("bulk_carrier_panamax_orderbook_oil")
                  ]
}

Forecast "bulk_carrier_panamax_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
        "01-01-2028"    0
    ]
    
    Extrapolate = FLAT
}

Fleet "container_8000_teu" {
    # vessel types
    Vessels = [Vessel("container_8000_teu_ice_oil")]
    
    # initial conditions
    InitialVessels = 4.81
    
    InitialSplit = [1.000
                    ]
    
    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.018, 0.025, 0.028, 0.026, 0.044, 0.083, 0.064, 0.080, 0.053, 0.064, 0.060, 0.045, 0.066, 0.065, 0.080, 0.035, 0.029, 0.018, 0.002, 0.001, 0.016, 0.016, 0.022, 0.049, 0.010]
    ]
    
    # trade growth
    TradeGrowth = 0 #Forecast("container_8000_teu_trade_growth")
    
    # orderbook
    Orderbooks = [Forecast("container_8000_teu_orderbook_oil")
                  ]
}


Forecast "container_8000_teu_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

Fleet "container_3500_teu" {
    # vessel types
    Vessels = [Vessel("container_3500_teu_ice_oil")]
    
    # initial conditions
    InitialVessels = 2.33
    
    InitialSplit = [1.000]
    
    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.033, 0.037, 0.032, 0.037, 0.056, 0.070, 0.086, 0.090, 0.058, 0.045, 0.028, 0.033, 0.031, 0.026, 0.023, 0.018, 0.022, 0.026, 0.028, 0.026, 0.027, 0.035, 0.057, 0.062, 0.013]
    ]
    
    # trade growth
    TradeGrowth = 0 #Forecast("container_3500_teu_trade_growth")

    # orderbook
    Orderbooks = [Forecast("container_3500_teu_orderbook_oil")
                  ]
}

Forecast "container_3500_teu_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}


Fleet "tanker_100k_dwt" {
    # vessel types
    Vessels = [Vessel("tanker_100k_dwt_ice_oil")]
    
    # initial conditions
    InitialVessels = 1.39
    
    InitialSplit = [1.000
                    ]
    
    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.008, 0.022, 0.054, 0.054, 0.068, 0.055, 0.062, 0.070, 0.084, 0.063, 0.056, 0.038, 0.020, 0.013, 0.022, 0.045, 0.053, 0.038, 0.038, 0.018, 0.028, 0.021, 0.015, 0.016, 0.039]
    ]
    
    # trade growth
    TradeGrowth = 0 #Forecast("tanker_100k_dwt_trade_growth")
    
    # orderbook
    Orderbooks = [Forecast("tanker_100k_dwt_orderbook_oil")
                  ]
}

Forecast "tanker_100k_dwt_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
        "01-01-2028"    0
    ]
    
    Extrapolate = FLAT
}


Fleet "tanker_35k_dwt" {
    # vessel types
    Vessels = [Vessel("tanker_35k_dwt_ice_oil")]
    
    # initial conditions
    InitialVessels = 2.92
    
    InitialSplit = [1.000
                    ]
    
    InitialAges = Array [
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.016, 0.014, 0.017, 0.017, 0.017, 0.016, 0.020, 0.030, 0.033, 0.036, 0.045, 0.053, 0.067, 0.063, 0.050, 0.046, 0.051, 0.044, 0.038, 0.035, 0.034, 0.033, 0.037, 0.033, 0.030, 0.031, 0.029, 0.021, 0.026, 0.019]
    ]
    
    # trade growth
    TradeGrowth = 0 #Forecast("tanker_35k_dwt_trade_growth")
    
    # orderbook
    Orderbooks = [Forecast("tanker_35k_dwt_orderbook_oil")
                  ]
}

Forecast "tanker_35k_dwt_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
        "01-01-2028"    0
        "01-01-2029"    0
    ]
    
    Extrapolate = FLAT
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
