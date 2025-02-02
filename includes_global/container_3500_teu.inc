# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Fleet "container_3500_teu" {	
	# vessel types
	Vessels = [Vessel("container_3500_teu_ice_oil"),
			   Vessel("container_3500_teu_ice_ammonia")]#,
#			   Vessel("container_3500_teu_ice_methane"),
#			   Vessel("container_3500_teu_ice_methanol")]
	
	# initial conditions
	InitialVessels = 3983
	
	InitialSplit = [1.000, #0.986
					0.000]#,
#					0.010,
#					0.004]
	
	InitialAges = Array [
		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]#,
#		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
#		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
	]
	
	InitialAgeSplit = Array [
		[0.033, 0.037, 0.032, 0.037, 0.056, 0.070, 0.086, 0.090, 0.058, 0.045, 0.028, 0.033, 0.031, 0.026, 0.023, 0.018, 0.022, 0.026, 0.028, 0.026, 0.027, 0.035, 0.057, 0.062, 0.013],
		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]#,
#		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.024, 0.000, 0.000, 0.000, 0.024, 0.024, 0.000, 0.098, 0.098, 0.049, 0.073, 0.024, 0.024, 0.317, 0.244],
#		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.067, 0.533, 0.400]
	]
	
	# trade growth
	TradeGrowth = Forecast("container_3500_teu_trade_growth")	

	# orderbook
	Orderbooks = [Forecast("container_3500_teu_orderbook_oil"),
				  Forecast("container_3500_teu_orderbook_ammonia")]#,
#				  Forecast("container_3500_teu_orderbook_methane"),
#			      Forecast("container_3500_teu_orderbook_methanol")]
	
	# energy efficiency
#	Efficiencies = [Efficiency("air_lubrication_container_3500_teu"),
#					Efficiency("exhaust_gas_boiler_ae_container_3500_teu"),
#					Efficiency("hull_painting_container_3500_teu"),
#					Efficiency("hull_shape_optimisation_container_3500_teu"),
#					Efficiency("lighting_system_container_3500_teu"),
#					Efficiency("load_optimization_container_3500_teu"),
#					Efficiency("propeller_cleaning_container_3500_teu"),
#					Efficiency("propeller_improvement_devices_container_3500_teu"),
#					Efficiency("rudder_autopilot_container_3500_teu"),
#					Efficiency("trim_draft_optimization_container_3500_teu"),
#					Efficiency("variable_frequency_drive_container_3500_teu"),
#					Efficiency("weather_routing_container_3500_teu")]
#
#	InitialEfficiencyShare = Array [
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.50, 0.30, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.50, 0.30, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.50, 0.30, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.50, 0.30, 0.50]
#	]
#
#	# alternative power
#	AlternativePowers = [AlternativePower("kite_container_3500_teu")]
#
#	InitialAlternativePowerShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# emission reduction
#	EmissionReductions = [EmissionReduction("calcium_looping_container_3500_teu")]
#
#	InitialEmissionReductionShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# fuel conversions
#	set_fuel_conversion_cost("*ice_oil*", "*ice_methanol*", 12.4e6)
#	set_fuel_conversion_cost("*ice_oil*", "*ice_ammonia*", 18.6e6)
#	set_fuel_conversion_cost("*ice_methane*", "*ice_ammonia*", 18.7e6)
}

Forecast "container_3500_teu_trade_growth" {
	Table [
		"01-01-2022"	0.0215
		"01-01-2023"	0.0215
		"01-01-2024"	0.0215
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

Forecast "container_3500_teu_orderbook_oil" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    16
		"01-01-2027"    16
	]
	
	Extrapolate = FLAT
}

Forecast "container_3500_teu_orderbook_ammonia" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    1
		"01-01-2027"    1
	]
	
	Extrapolate = FLAT
}

Forecast "container_3500_teu_orderbook_methane" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    2
		"01-01-2027"    4
	]
	
	Extrapolate = FLAT
}

Forecast "container_3500_teu_orderbook_methanol" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    12
		"01-01-2027"    12
	]
	
	Extrapolate = FLAT
}
