# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Fleet "container_15000_teu" {	
	# vessel types
	Vessels = [Vessel("container_15000_teu_ice_oil"),
			   Vessel("container_15000_teu_ice_ammonia"),
#			   Vessel("container_15000_teu_ice_liquid_hydrogen"),
               Vessel("container_15000_teu_ice_hydrogen"),
			   Vessel("container_15000_teu_ice_methanol"),
               Vessel("container_15000_teu_ice_diesel")]

	# initial conditions
	InitialVessels = 808
	
	InitialSplit = [1.000, # 0.817
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
		[0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.008, 0.003, 0.008, 0.038, 0.053, 0.065, 0.052, 0.064, 0.068, 0.044, 0.065, 0.079, 0.073, 0.042, 0.041, 0.030, 0.097, 0.118, 0.050],
		[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
	]

	# trade growth
	TradeGrowth = Forecast("container_15000_teu_trade_growth")
	
	# orderbook
	Orderbooks = [Forecast("container_15000_teu_orderbook_oil"),
			      Forecast("container_15000_teu_orderbook_ammonia"),
				  Forecast("container_15000_teu_orderbook_liquid_hydrogen"),
				  Forecast("container_15000_teu_orderbook_methanol"),
                  Forecast("container_15000_teu_orderbook_diesel")]
	
	# energy efficiency
#	Efficiencies = [Efficiency("air_lubrication_container_15000_teu"),
#					Efficiency("exhaust_gas_boiler_ae_container_15000_teu"),
#					Efficiency("hull_painting_container_15000_teu"),
#					Efficiency("hull_shape_optimisation_container_15000_teu"),
#					Efficiency("lighting_system_container_15000_teu"),
#					Efficiency("load_optimization_container_15000_teu"),
#					Efficiency("propeller_cleaning_container_15000_teu"),
#					Efficiency("propeller_improvement_devices_container_15000_teu"),
#					Efficiency("rudder_autopilot_container_15000_teu"),
#					Efficiency("shaft_motor_container_15000_teu"),
#					Efficiency("trim_draft_optimization_container_15000_teu"),
#					Efficiency("variable_frequency_drive_container_15000_teu"),
#					Efficiency("waste_heat_recovery_system_container_15000_teu"),
#					Efficiency("weather_routing_container_15000_teu")]
					
#	InitialEfficiencyShare = Array [
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.1, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.1, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.1, 0.50]
#		[0.0, 0.125, 0.35, 0.2, 0.50, 0.75, 0.25, 0.25, 0.75, 0.1, 0.50, 0.30, 0.1, 0.50]
#	]
		
#	# alternative power
#	AlternativePowers = [AlternativePower("kite_container_15000_teu")]
#
#	InitialAlternativePowerShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# emission reduction
#	EmissionReductions = [EmissionReduction("chemical_absorption_container_15000_teu")]
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
#	set_fuel_conversion_cost("*ice_oil*", "*ice_methanol*", 24.3e6)
#	set_fuel_conversion_cost("*ice_oil*", "*ice_ammonia*", 36.4e6)
#	set_fuel_conversion_cost("*ice_methane*", "*ice_ammonia*", 25.9e6)

    # 5% discount rate
    DiscountRate = 0.05
}

Forecast "container_15000_teu_trade_growth" {
	Table [
		"01-01-2022"	0.0515
		"01-01-2023"	0.0515
		"01-01-2024"	0.0515
		"01-01-2025"	0.0459
		"01-01-2026"	0.0459
		"01-01-2027"	0.0459
		"01-01-2028"	0.0459
		"01-01-2029"	0.0459
		"01-01-2030"	0.0314
		"01-01-2031"	0.0314
		"01-01-2032"	0.0314
		"01-01-2033"	0.0314
		"01-01-2034"	0.0314
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

Forecast "container_15000_teu_orderbook_oil" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    9
		"01-01-2027"    9
	]
	
	Extrapolate = FLAT
}

Forecast "container_15000_teu_orderbook_ammonia" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    0
		"01-01-2027"    0
	]
	
	Extrapolate = FLAT
}

Forecast "container_15000_teu_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

# DMM: This needs to be checked/updated
Forecast "container_15000_teu_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
    ]
    
    Extrapolate = FLAT
}

Forecast "container_15000_teu_orderbook_methane" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    10
		"01-01-2027"    17
		"01-01-2028"    18
	]
	
	Extrapolate = FLAT
}

Forecast "container_15000_teu_orderbook_methanol" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    28
		"01-01-2027"    46
		"01-01-2028"    50
	]
	
	Extrapolate = FLAT
}
