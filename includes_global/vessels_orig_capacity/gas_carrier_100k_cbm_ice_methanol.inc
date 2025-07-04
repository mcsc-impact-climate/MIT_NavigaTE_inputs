# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.


Vessel "gas_carrier_100k_cbm_ice_methanol" {
	
	PropulsionLoadLevel = Curve("speed_power_curve_gas_carrier_100k_cbm")
	ElectricalLoadLevelAtSea = 0.6
	ElectricalLoadLevelInPort = 2.4
	HeatLoadLevelAtSea = 0.6	
	HeatLoadLevelInPort = 0.6	
	
	PowerSystem = PowerSystem("ice_methanol_gas_carrier_100k_cbm")
	Tanks = [Tank("main_methanol_gas_carrier_100k_cbm"), Tank("pilot_oil_gas_carrier_100k_cbm")]
	
	Route = Route("globalized_gas_carrier_100k_cbm")
	
	NominalCapacity = 100000
	

	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 129e6
	OPEX = 2.9e6
}