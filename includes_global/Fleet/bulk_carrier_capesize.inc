# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Fleet "bulk_carrier_capesize" {	
	# vessel types
	Vessels = [Vessel("bulk_carrier_capesize_ice_oil"),
			   Vessel("bulk_carrier_capesize_ice_ammonia"),
#			   Vessel("bulk_carrier_capesize_ice_liquid_hydrogen"),
               Vessel("bulk_carrier_capesize_ice_hydrogen"),
			   Vessel("bulk_carrier_capesize_ice_methanol"),
               Vessel("bulk_carrier_capesize_ice_diesel")]
	
	# initial conditions
	InitialVessels = 2067
	
    InitialSplit = [1.000,#0.974,
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
        [0.008, 0.005, 0.013, 0.020, 0.023, 0.029, 0.028, 0.022, 0.055, 0.105, 0.124, 0.107, 0.051, 0.047, 0.044, 0.051, 0.037, 0.025, 0.041, 0.056, 0.043, 0.021, 0.017, 0.012, 0.015], 
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
    ]
	
	# trade growth
	TradeGrowth = Forecast("bulk_carrier_capesize_trade_growth")
	
	# orderbook
	Orderbooks = [Forecast("bulk_carrier_capesize_orderbook_oil"),
				  Forecast("bulk_carrier_capesize_orderbook_ammonia"),
				  Forecast("bulk_carrier_capesize_orderbook_liquid_hydrogen"),
				  Forecast("bulk_carrier_capesize_orderbook_methanol"),
                  Forecast("bulk_carrier_capesize_orderbook_diesel")]
	
	# energy efficiency
#	Efficiencies = [Efficiency("air_lubrication_bulk_carrier_capesize"),
#					Efficiency("exhaust_gas_boiler_ae_bulk_carrier_capesize"),
#					Efficiency("hull_painting_bulk_carrier_capesize"),
#					Efficiency("hull_shape_optimisation_bulk_carrier_capesize"),
#					Efficiency("lighting_system_bulk_carrier_capesize"),
#					Efficiency("propeller_cleaning_bulk_carrier_capesize"),
#					Efficiency("propeller_improvement_devices_bulk_carrier_capesize"),
#					Efficiency("rudder_autopilot_bulk_carrier_capesize"),
#					Efficiency("shaft_motor_bulk_carrier_capesize"),
#					Efficiency("trim_draft_optimization_bulk_carrier_capesize"),
#					Efficiency("variable_frequency_drive_bulk_carrier_capesize"),
#					Efficiency("waste_heat_recovery_system_bulk_carrier_capesize"),
#					Efficiency("weather_routing_bulk_carrier_capesize")]
#
#	InitialEfficiencyShare = Array [
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.0, 0.5, 0.25, 0.0, 0.5]
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.0, 0.5, 0.25, 0.0, 0.5]
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.0, 0.5, 0.25, 0.0, 0.5]
#		[0.0, 0.125, 0.1, 0.2, 0.5, 0.5, 0.25, 0.5, 0.0, 0.5, 0.25, 0.0, 0.5]
#	]
#
#	# alternative power
#	AlternativePowers = [AlternativePower("flettner_rotor_bulk_carrier_capesize")]
#
#	InitialAlternativePowerShare = Array [
#		[0]
#		[0]
#		[0]
#		[0]
#	]
#
#	# emission reduction
#	EmissionReductions = [EmissionReduction("chemical_absorption_bulk_carrier_capesize")]
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
#	set_fuel_conversion_cost("*ice_oil*", "*ice_methanol*", 25.4e6)
#	set_fuel_conversion_cost("*ice_oil*", "*ice_ammonia*", 33.1e6)
#	set_fuel_conversion_cost("*ice_methane*", "*ice_ammonia*", 17.0e6)

    # 5% discount rate
    DiscountRate = 0.05
}

Forecast "bulk_carrier_capesize_trade_growth" {
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

Forecast "bulk_carrier_capesize_orderbook_oil" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    20
		"01-01-2027"    29
		"01-01-2028"    34
	]
	
	Extrapolate = FLAT
}

Forecast "bulk_carrier_capesize_orderbook_ammonia" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    4
		"01-01-2027"    9
		"01-01-2028"    10
	]
	
	Extrapolate = FLAT
}

Forecast "bulk_carrier_capesize_orderbook_liquid_hydrogen" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
        "01-01-2028"    0
    ]
    
    Extrapolate = FLAT
}

# DMM: This needs to be checked/updated
Forecast "bulk_carrier_capesize_orderbook_diesel" {
    Table [
        "01-01-2025"    0
        "01-01-2026"    0
        "01-01-2027"    0
        "01-01-2028"    0
    ]
    
    Extrapolate = FLAT
}

Forecast "bulk_carrier_capesize_orderbook_methane" {
	Table [
		"01-01-2025"	0
		"01-01-2026"	0
		"01-01-2027"	0
	]
	
	Extrapolate = FLAT
}

Forecast "bulk_carrier_capesize_orderbook_methanol" {
	Table [
		"01-01-2025"	0
		"01-01-2026"    1
		"01-01-2027"    2
	]
	
	Extrapolate = FLAT
}
