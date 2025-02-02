# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Fleet "gas_carrier_100k_cbm" {
	# vessel types
	Vessels = [Vessel("gas_carrier_100k_cbm_ice_oil"),
			   Vessel("gas_carrier_100k_cbm_ice_ammonia")]#,
#			   Vessel("gas_carrier_100k_cbm_ice_methane"),
#			   Vessel("gas_carrier_100k_cbm_ice_methanol")]
	
	# initial conditions
	InitialVessels = 2156
	
	InitialSplit = [1.000,#0.623,
					0.000]#,
#					0.377,
#					0.000]
	
	InitialAges = Array [
		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]#,
#		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
#		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
	]
	
	InitialAgeSplit = Array [
		[0.023, 0.019, 0.025, 0.016, 0.013, 0.031, 0.048, 0.080, 0.060, 0.051, 0.045, 0.033, 0.037, 0.038, 0.063, 0.067, 0.052, 0.031, 0.031, 0.039, 0.038, 0.040, 0.060, 0.038, 0.025],
		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]#,
#		[0.001, 0.011, 0.017, 0.025, 0.020, 0.032, 0.034, 0.036, 0.026, 0.025, 0.014, 0.002, 0.022, 0.044, 0.037, 0.042, 0.044, 0.069, 0.053, 0.048, 0.079, 0.043, 0.048, 0.111, 0.117],
#		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
	]
	
	# trade growth
	TradeGrowth = Forecast("gas_carrier_100k_cbm_trade_growth")
	
	# orderbooks
	Orderbooks = [Forecast("gas_carrier_100k_cbm_orderbook_oil"),
				  Forecast("gas_carrier_100k_cbm_orderbook_ammonia")]#,
#				  Forecast("gas_carrier_100k_cbm_orderbook_methane"),
#				  Forecast("gas_carrier_100k_cbm_orderbook_methanol")]
	
	# energy efficiency
#	Efficiencies = [Efficiency("air_lubrication_gas_carrier_100k_cbm"),
#					Efficiency("cargo_systems_gas_carrier_100k_cbm"),
#					Efficiency("exhaust_gas_boiler_ae_gas_carrier_100k_cbm"),
#					Efficiency("hull_painting_gas_carrier_100k_cbm"),
#					Efficiency("hull_shape_optimisation_gas_carrier_100k_cbm"),
#					Efficiency("lighting_system_gas_carrier_100k_cbm"),
#					Efficiency("load_optimization_gas_carrier_100k_cbm"),
#					Efficiency("propeller_cleaning_gas_carrier_100k_cbm"),
#					Efficiency("propeller_improvement_devices_gas_carrier_100k_cbm"),
#					Efficiency("rudder_autopilot_gas_carrier_100k_cbm"),
#					Efficiency("steam_system_efficiency_gas_carrier_100k_cbm"),
#					Efficiency("trim_draft_optimization_gas_carrier_100k_cbm"),
#					Efficiency("variable_frequency_drive_gas_carrier_100k_cbm"),
#					Efficiency("weather_routing_gas_carrier_100k_cbm")]
#	
#	InitialEfficiencyShare = Array [
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.25, 0.25, 0.5, 0.75, 0.50, 1.0, 0.5]
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.25, 0.25, 0.5, 0.75, 0.50, 1.0, 0.5]
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.25, 0.25, 0.5, 0.75, 0.50, 1.0, 0.5]
#		[0.0, 0.0, 0.125, 0.15, 0.2, 0.5, 0.20, 0.25, 0.25, 0.5, 0.75, 0.50, 1.0, 0.5]
#	]
#
#	# alternative power
#	AlternativePowers = [AlternativePower("flettner_rotor_gas_carrier_100k_cbm")]
#	
#	InitialAlternativePowerShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#	
#	# emission reduction
#	EmissionReductions = [EmissionReduction("chemical_absorption_gas_carrier_100k_cbm")]
#	
#	InitialEmissionReductionShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
}

Forecast "gas_carrier_100k_cbm_trade_growth" {
	Table [
		"01-01-2022"	0.07574
		"01-01-2023"	0.07574
		"01-01-2024"	0.07574
		"01-01-2025"	0.02654
		"01-01-2026"	0.02654
		"01-01-2027"	0.02654
		"01-01-2028"	0.02654
		"01-01-2029"	0.02654
		"01-01-2030"	-0.00526
		"01-01-2031"	-0.00526
		"01-01-2032"	-0.00526
		"01-01-2033"	-0.00526
		"01-01-2034"	-0.00526
		"01-01-2035"	-0.00507
		"01-01-2036"	-0.00507
		"01-01-2037"	-0.00507
		"01-01-2038"	-0.00507
		"01-01-2039"	-0.00507
		"01-01-2040"	-0.00912
		"01-01-2041"	-0.00912
		"01-01-2042"	-0.00912
		"01-01-2043"	-0.00912
		"01-01-2044"	-0.00912
		"01-01-2045"	-0.00883
		"01-01-2046"	-0.00883
		"01-01-2047"	-0.00883
		"01-01-2048"	-0.00883
		"01-01-2049"	-0.00883
		"01-01-2050"	-0.00883
	]
}

Forecast "gas_carrier_100k_cbm_orderbook_oil" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    59
		"01-01-2027"    98
		"01-01-2028"    105
	]
	
	Extrapolate = FLAT
}

Forecast "gas_carrier_100k_cbm_orderbook_ammonia" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    4
		"01-01-2027"    6
		"01-01-2028"    8
	]
	
	Extrapolate = FLAT
}

Forecast "gas_carrier_100k_cbm_orderbook_methane" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    89
		"01-01-2027"    148
		"01-01-2028"    174
		"01-01-2029"    177
	]
	
	Extrapolate = FLAT
}

Forecast "gas_carrier_100k_cbm_orderbook_methanol" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    0
		"01-01-2027"    0
	]
	
	Extrapolate = FLAT
}
