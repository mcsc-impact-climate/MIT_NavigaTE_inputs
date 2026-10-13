Fleet "bulk_carrier" {
    Vessels = [
               Vessel("bulk_carrier_handy_ice_oil"),
               Vessel("bulk_carrier_panamax_ice_oil"),
               Vessel("bulk_carrier_capesize_ice_oil")
               ]
    # 1 bulk_carrier_handy ; 4 bulk_carrier_panamax ; 2 bulk_carrier_capesize
    InitialVessels = 1.15
    
    InitialSplit = [0.15652173913043478261,
                    0.55652173913043478261,
                    0.28695652173913043478
                    ]

    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    InitialAgeSplit = Array [
        [0.020, 0.017, 0.015, 0.018, 0.022, 0.020, 0.025, 0.030, 0.051, 0.084, 0.095, 0.092, 0.057, 0.046, 0.058, 0.047, 0.039, 0.024, 0.029, 0.030, 0.031, 0.031, 0.035, 0.046, 0.039],
        [0.033, 0.017, 0.008, 0.025, 0.029, 0.034, 0.026, 0.024, 0.025, 0.055, 0.085, 0.105, 0.077, 0.050, 0.039, 0.037, 0.035, 0.022, 0.045, 0.047, 0.034, 0.038, 0.040, 0.034, 0.036],
        [0.008, 0.005, 0.013, 0.020, 0.023, 0.029, 0.028, 0.022, 0.055, 0.105, 0.124, 0.107, 0.051, 0.047, 0.044, 0.051, 0.037, 0.025, 0.041, 0.056, 0.043, 0.021, 0.017, 0.012, 0.015]
    ]

    # trade growth
    TradeGrowth = 0
    
    # orderbook. DME: Setting all orderbooks empty until I can come up with actual orderbook estimates on the S-R corridor
    Orderbooks = [
                  Forecast("bulk_carrier_handy_orderbook_oil"),
                  Forecast("bulk_carrier_panamax_orderbook_oil"),
                  Forecast("bulk_carrier_capesize_orderbook_oil")
                  ]
}

Fleet "container" {
    Vessels = [
               Vessel("container_15000_teu_ice_oil"),
               Vessel("container_8000_teu_ice_oil"),
               Vessel("container_3500_teu_ice_oil")
               ]

    InitialVessels = 16.3

    InitialSplit = [0.843251,
                    0.156749,
                    0.000
                    ]

    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]

    InitialAgeSplit = Array [
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.022, 0.000, 0.067, 0.056, 0.122, 0.033, 0.089, 0.167, 0.067, 0.444, 0.067, 0.089, 0.011, 0.067, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.120, 0.200, 0.200, 0.120, 0.040, 0.000, 0.120, 0.120, 0.040, 0.000, 0.000, 0.000, 0.040, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.333, 0.000, 0.000, 0.000, 0.000, 0.333, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.333, 0.000, 0.000]
    ]

    # trade growth
    TradeGrowth = 0

    # orderbook. DME: Setting all orderbooks empty until I can come up with actual orderbook estimates on the S-R corridor
    Orderbooks = [
                  Forecast("container_15000_teu_orderbook_oil"),
                  Forecast("container_8000_teu_orderbook_oil"),
                  Forecast("container_3500_teu_orderbook_oil")
                  ]
}

Fleet "tanker" {
    Vessels = [
               Vessel("tanker_300k_dwt_ice_oil"),
               Vessel("tanker_100k_dwt_ice_oil"),
               Vessel("tanker_35k_dwt_ice_oil")
               ]

    InitialVessels = 3.2

    InitialSplit = [0.050902,
                    0.408014,
                    0.541084
                    ]

    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]

    InitialAgeSplit = Array [
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.000, 0.000, 0.000, 0.333, 0.000, 0.333, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    ]

    # trade growth
    TradeGrowth = 0

    # orderbook. DME: Setting all orderbooks empty until I can come up with actual orderbook estimates on the S-R corridor
    Orderbooks = [
                  Forecast("tanker_300k_dwt_orderbook_oil"),
                  Forecast("tanker_100k_dwt_orderbook_oil"),
                  Forecast("tanker_35k_dwt_orderbook_oil")
                  ]
}

Fleet "fleet_gas" {
    Vessels = [
               Vessel("gas_carrier_100k_cbm_ice_oil")
              ]

    InitialVessels = 0.04

    InitialSplit = [1.000
                    ]

    InitialAges = Array [
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]

    InitialAgeSplit = Array [
        [0.023, 0.019, 0.025, 0.016, 0.013, 0.031, 0.048, 0.080, 0.060, 0.051, 0.045, 0.033, 0.037, 0.038, 0.063, 0.067, 0.052, 0.031, 0.031, 0.039, 0.038, 0.040, 0.060, 0.038, 0.025]
    ]

    # trade growth
    TradeGrowth = 0

    # orderbook. DME: Setting all orderbooks empty until I can come up with actual orderbook estimates on the S-R corridor
    Orderbooks = [
                  Forecast("gas_carrier_100k_cbm_orderbook_oil")
                  ]
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

## Panamax bulk carrier ##
Forecast "bulk_carrier_panamax_orderbook_oil" {
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

## 15000 TEU container ##
Forecast "container_15000_teu_orderbook_oil" {
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

## 3500 TEU container ##
Forecast "container_3500_teu_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## 300k DWT tanker ##
Forecast "tanker_300k_dwt_orderbook_oil" {
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

## 35k DWT tanker ##
Forecast "tanker_35k_dwt_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}

## 100k CBM gas carrier ##
Forecast "gas_carrier_100k_cbm_orderbook_oil" {
    Table [
        "01-01-2025"    0
        "01-01-2050"    0
    ]

    Extrapolate = FLAT
}
####################################