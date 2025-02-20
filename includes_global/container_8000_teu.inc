# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Fleet "container_8000_teu" {	
	# vessel types
	Vessels = [Vessel("container_8000_teu_ice_oil"),
			   Vessel("container_8000_teu_ice_ammonia"),
			   Vessel("container_8000_teu_ice_liquid_hydrogen"),
			   Vessel("container_8000_teu_ice_methanol"),
               Vessel("container_8000_teu_ice_diesel")]
	
	# initial conditions
	InitialVessels = 1043
	
	InitialSplit = [1.000,#0.933,
					0.000,
                    0.000,
					0.000,
					0.000]
	
	InitialAges = Array [
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
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
	]
	
	# trade growth
	TradeGrowth = Forecast("container_8000_teu_trade_growth")	
	
	# orderbook
	Orderbooks = [Forecast("container_8000_teu_orderbook_oil"),
				  Forecast("container_8000_teu_orderbook_ammonia"),
				  Forecast("container_8000_teu_orderbook_liquid_hydrogen"),
				  Forecast("container_8000_teu_orderbook_methanol"),
                  Forecast("container_8000_teu_orderbook_diesel")]
	
	# energy efficiency
#	Efficiencies = [Efficiency("air_lubrication_container_8000_teu"),
#					Efficiency("exhaust_gas_boiler_ae_container_8000_teu"),
#					Efficiency("hull_painting_container_8000_teu"),
#					Efficiency("hull_shape_optimisation_container_8000_teu"),
#					Efficiency("lighting_system_container_8000_teu"),
#					Efficiency("load_optimization_container_8000_teu"),
#					Efficiency("propeller_cleaning_container_8000_teu"),
#					Efficiency("propeller_improvement_devices_container_8000_teu"),
#					Efficiency("rudder_autopilot_container_8000_teu"),
#					Efficiency("shaft_motor_container_8000_teu"),
#					Efficiency("trim_draft_optimization_container_8000_teu"),
#					Efficiency("variable_frequency_drive_container_8000_teu"),
#					Efficiency("waste_heat_recovery_system_container_8000_teu"),
#					Efficiency("weather_routing_container_8000_teu")]
#					
#	InitialEfficiencyShare = Array [
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.10, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.10, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.10, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.10, 0.50]
#	]
#	
#	# alternative power
#	AlternativePowers = [AlternativePower("kite_container_8000_teu")]
#	
#	InitialAlternativePowerShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#	
#	# emission reduction
#	EmissionReductions = [EmissionReduction("chemical_absorption_container_8000_teu")]
#	
#	InitialEmissionReductionShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#	
#	# technology compatibility
#	set_technology_compatibility("shaft_motor*", "waste_heat_recovery_system*", 0)
#	
#	# fuel conversions
#	set_fuel_conversion_cost("*ice_oil*", "*ice_methanol*", 17.1e6)
#	set_fuel_conversion_cost("*ice_oil*", "*ice_ammonia*", 25.6e6)
#	set_fuel_conversion_cost("*ice_methane*", "*ice_ammonia*", 21.5e6)

    # 5% discount rate
    DiscountRate = 0.05
}

Forecast "container_8000_teu_trade_growth" {
	Table [
		"01-01-2022"	0.0234
		"01-01-2023"	0.0234
		"01-01-2024"	0.0234
		"01-01-2025"	0.0220
		"01-01-2026"	0.0220
		"01-01-2027"	0.0220
		"01-01-2028"	0.0220
		"01-01-2029"	0.0220
		"01-01-2030"	0.0220
		"01-01-2031"	0.0220
		"01-01-2032"	0.0220
		"01-01-2033"	0.0220
		"01-01-2034"	0.0220
		"01-01-2035"	0.0194
		"01-01-2036"	0.0194
		"01-01-2037"	0.0194
		"01-01-2038"	0.0194
		"01-01-2039"	0.0194
		"01-01-2040"	0.0168
		"01-01-2041"	0.0168
		"01-01-2042"	0.0168
		"01-01-2043"	0.0168
		"01-01-2044"	0.0168
		"01-01-2045"	0.0153
		"01-01-2046"	0.0153
		"01-01-2047"	0.0153
		"01-01-2048"	0.0153
		"01-01-2049"	0.0153
		"01-01-2050"	0.0153
	]
}

Forecast "container_8000_teu_orderbook_oil" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    6
		"01-01-2027"    7
	]
	
	Extrapolate = FLAT
}

Forecast "container_8000_teu_orderbook_ammonia" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    0
		"01-01-2027"    0
	]
	
	Extrapolate = FLAT
}

Forecast "container_8000_teu_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

# DMM: This needs to be checked/updated
Forecast "container_8000_teu_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

Forecast "container_8000_teu_orderbook_methane" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    17
		"01-01-2027"    31
	]
	
	Extrapolate = FLAT
}

Forecast "container_8000_teu_orderbook_methanol" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    11
		"01-01-2027"    17
	]
	
	Extrapolate = FLAT
}
