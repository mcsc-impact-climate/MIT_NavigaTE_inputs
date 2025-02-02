# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Fleet "bulk_carrier_panamax" {
	# vessel types
	Vessels = [Vessel("bulk_carrier_panamax_ice_oil"),
			   Vessel("bulk_carrier_panamax_ice_ammonia")]#,
#			   Vessel("bulk_carrier_panamax_ice_methane"),
#			   Vessel("bulk_carrier_panamax_ice_methanol")]
	
	# initial conditions
	InitialVessels = 3110
	
	InitialSplit = [1.000,#0.999,
					0.000]#,
#					0.001,
#					0.000]
	
	InitialAges = Array [
		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]#,
#		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
#		[24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
	]
	
	InitialAgeSplit = Array [
		[0.033, 0.017, 0.008, 0.025, 0.029, 0.034, 0.026, 0.024, 0.025, 0.055, 0.085, 0.105, 0.077, 0.050, 0.039, 0.037, 0.035, 0.022, 0.045, 0.047, 0.034, 0.038, 0.040, 0.034, 0.036],
		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]#,
#		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.667, 0.000, 0.333],
#		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000]
	]

	# trade growth
	TradeGrowth = Forecast("bulk_carrier_panamax_trade_growth")

	# orderbook
	Orderbooks = [Forecast("bulk_carrier_panamax_orderbook_oil"),
				  Forecast("bulk_carrier_panamax_orderbook_ammonia")]#,
#				  Forecast("bulk_carrier_panamax_orderbook_methane"),
#				  Forecast("bulk_carrier_panamax_orderbook_methanol")]
	
	# energy efficiency
#	Efficiencies = [Efficiency("air_lubrication_bulk_carrier_panamax"),
#					Efficiency("exhaust_gas_boiler_ae_bulk_carrier_panamax"),
#					Efficiency("hull_painting_bulk_carrier_panamax"),
#					Efficiency("hull_shape_optimisation_bulk_carrier_panamax"),
#					Efficiency("lighting_system_bulk_carrier_panamax"),
#					Efficiency("propeller_cleaning_bulk_carrier_panamax"),
#					Efficiency("propeller_improvement_devices_bulk_carrier_panamax"),
#					Efficiency("rudder_autopilot_bulk_carrier_panamax"),
#					Efficiency("trim_draft_optimization_bulk_carrier_panamax"),
#					Efficiency("variable_frequency_drive_bulk_carrier_panamax"),
#					Efficiency("weather_routing_bulk_carrier_panamax")]
#
#	InitialEfficiencyShare = Array [
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5]
#	]
#
#	# alternative power
#	AlternativePowers = [AlternativePower("flettner_rotor_bulk_carrier_panamax")]
#
#	InitialAlternativePowerShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# emission reduction
#	EmissionReductions = [EmissionReduction("chemical_absorption_bulk_carrier_panamax")]
#
#	InitialEmissionReductionShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# fuel conversions
#	set_fuel_conversion_cost("*ice_oil*", "*ice_methanol*", 17.2e6)
#	set_fuel_conversion_cost("*ice_oil*", "*ice_ammonia*", 23.7e6)
#	set_fuel_conversion_cost("*ice_methane*", "*ice_ammonia*", 14.1e6)
}

Forecast "bulk_carrier_panamax_trade_growth" {
	Table [
		"01-01-2022"	0.0248
		"01-01-2023"	0.0248
		"01-01-2024"	0.0248
		"01-01-2025"	0.0158
		"01-01-2026"	0.0158
		"01-01-2027"	0.0158
		"01-01-2028"	0.0158
		"01-01-2029"	0.0158
		"01-01-2030"	0.0096
		"01-01-2031"	0.0096
		"01-01-2032"	0.0096
		"01-01-2033"	0.0096
		"01-01-2034"	0.0096
		"01-01-2035"	0.0087
		"01-01-2036"	0.0087
		"01-01-2037"	0.0087
		"01-01-2038"	0.0087
		"01-01-2039"	0.0087
		"01-01-2040"	0.0071
		"01-01-2041"	0.0071
		"01-01-2042"	0.0071
		"01-01-2043"	0.0071
		"01-01-2044"	0.0071
		"01-01-2045"	0.0070
		"01-01-2046"	0.0070
		"01-01-2047"	0.0070
		"01-01-2048"	0.0070
		"01-01-2049"	0.0070
		"01-01-2050"	0.0070
	]
}


Forecast "bulk_carrier_panamax_orderbook_oil" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    109
		"01-01-2027"    127
		"01-01-2028"    129
	]
	
	Extrapolate = FLAT
}

Forecast "bulk_carrier_panamax_orderbook_ammonia" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    0
		"01-01-2027"    0
	]
	
	Extrapolate = FLAT
}

Forecast "bulk_carrier_panamax_orderbook_methane" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    1
		"01-01-2027"    1
	]
	
	Extrapolate = FLAT
}

Forecast "bulk_carrier_panamax_orderbook_methanol" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    3
		"01-01-2027"    7
		"01-01-2028"    8
	]
	
	Extrapolate = FLAT
}
