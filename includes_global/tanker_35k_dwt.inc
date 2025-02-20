# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Fleet "tanker_35k_dwt" {
	# vessel types
	Vessels = [Vessel("tanker_35k_dwt_ice_oil"),
			   Vessel("tanker_35k_dwt_ice_ammonia"),
			   Vessel("tanker_35k_dwt_ice_liquid_hydrogen"),
			   Vessel("tanker_35k_dwt_ice_methanol"),
               Vessel("tanker_35k_dwt_ice_diesel")]
	
	# initial conditions
	InitialVessels = 9770
	
	InitialSplit = [1.000,      #0.987,
					0.000,
                    0.000,
					0.000,
					0.000]
	
	InitialAges = Array [
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
	]
	
	InitialAgeSplit = Array [
		[0.016, 0.014, 0.017, 0.017, 0.017, 0.016, 0.020, 0.030, 0.033, 0.036, 0.045, 0.053, 0.067, 0.063, 0.050, 0.046, 0.051, 0.044, 0.038, 0.035, 0.034, 0.033, 0.037, 0.033, 0.030, 0.031, 0.029, 0.021, 0.026, 0.019],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
	]
	
	# trade growth
	TradeGrowth = Forecast("tanker_35k_dwt_trade_growth")
	
	# orderbook
	Orderbooks = [Forecast("tanker_35k_dwt_orderbook_oil"),
				  Forecast("tanker_35k_dwt_orderbook_ammonia"),
				  Forecast("tanker_35k_dwt_orderbook_liquid_hydrogen"),
				  Forecast("tanker_35k_dwt_orderbook_methanol"),
                  Forecast("tanker_35k_dwt_orderbook_diesel")]
		
	# energy efficiency	
#	Efficiencies = [Efficiency("air_lubrication_tanker_35k_dwt"),
#					Efficiency("cargo_systems_tanker_35k_dwt"),
#					Efficiency("exhaust_gas_boiler_ae_tanker_35k_dwt"),
#					Efficiency("hull_painting_tanker_35k_dwt"),
#					Efficiency("hull_shape_optimisation_tanker_35k_dwt"),
#					Efficiency("lighting_system_tanker_35k_dwt"),
#					Efficiency("load_optimization_tanker_35k_dwt"),
#					Efficiency("propeller_cleaning_tanker_35k_dwt"),
#					Efficiency("propeller_improvement_devices_tanker_35k_dwt"),
#					Efficiency("rudder_autopilot_tanker_35k_dwt"),
#					Efficiency("steam_system_efficiency_tanker_35k_dwt"),
#					Efficiency("trim_draft_optimization_tanker_35k_dwt"),
#					Efficiency("variable_frequency_drive_tanker_35k_dwt"),
#					Efficiency("weather_routing_tanker_35k_dwt")]
#
#	InitialEfficiencyShare = Array [
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.5, 0.25, 0.5, 0.75, 0.50, 0.25, 0.5]
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.5, 0.25, 0.5, 0.75, 0.50, 0.25, 0.5]
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.5, 0.25, 0.5, 0.75, 0.50, 0.25, 0.5]
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.5, 0.25, 0.5, 0.75, 0.50, 0.25, 0.5]
#	]
#
#	# alternative power
#	AlternativePowers = [AlternativePower("flettner_rotor_tanker_35k_dwt")]
#
#	InitialAlternativePowerShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# emission reduction
#	EmissionReductions = [EmissionReduction("calcium_looping_tanker_35k_dwt")]
#
#	InitialEmissionReductionShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# fuel conversions
#	set_fuel_conversion_cost("*ice_oil*", "*ice_methanol*", 11.4e6)
#	set_fuel_conversion_cost("*ice_oil*", "*ice_ammonia*", 17.3e6)
#	set_fuel_conversion_cost("*ice_methane*", "*ice_ammonia*", 17.3e6)

    # 5% discount rate
    DiscountRate = 0.05
}

Forecast "tanker_35k_dwt_trade_growth" {
	Table [
		"01-01-2022"	0.00957
		"01-01-2023"	0.00957
		"01-01-2024"	0.00957
		"01-01-2025"	0.00172
		"01-01-2026"	0.00172
		"01-01-2027"	0.00172
		"01-01-2028"	0.00172
		"01-01-2029"	0.00172
		"01-01-2030"	-0.00733
		"01-01-2031"	-0.00733
		"01-01-2032"	-0.00733
		"01-01-2033"	-0.00733
		"01-01-2034"	-0.00733
		"01-01-2035"	-0.00684
		"01-01-2036"	-0.00684
		"01-01-2037"	-0.00684
		"01-01-2038"	-0.00684
		"01-01-2039"	-0.00684
		"01-01-2040"	-0.00629
		"01-01-2041"	-0.00629
		"01-01-2042"	-0.00629
		"01-01-2043"	-0.00629
		"01-01-2044"	-0.00629
		"01-01-2045"	-0.00559
		"01-01-2046"	-0.00559
		"01-01-2047"	-0.00559
		"01-01-2048"	-0.00559
		"01-01-2049"	-0.00559
		"01-01-2050"	-0.00559
	]
}

Forecast "tanker_35k_dwt_orderbook_oil" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    138
		"01-01-2027"    188
		"01-01-2028"    195
		"01-01-2029"    196
	]
	
	Extrapolate = FLAT
}

Forecast "tanker_35k_dwt_orderbook_ammonia" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    0
		"01-01-2027"    0
	]
	
	Extrapolate = FLAT
}

# DMM: This needs to be checked/updated
Forecast "tanker_35k_dwt_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

Forecast "tanker_35k_dwt_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

Forecast "tanker_35k_dwt_orderbook_methane" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    5
		"01-01-2027"    7
	]
	
	Extrapolate = FLAT
}

Forecast "tanker_35k_dwt_orderbook_methanol" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    11
		"01-01-2027"    11
	]
	
	Extrapolate = FLAT
}
