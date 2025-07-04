# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Vessel "container_8000_teu_ice_ammonia" {
	
	PropulsionLoadLevel = Curve("speed_power_curve_container_8000_teu")
	ElectricalLoadLevelAtSea = 2.0
	ElectricalLoadLevelInPort = 1.6
	HeatLoadLevelAtSea = 0.3
	HeatLoadLevelInPort = 1.43
	
	PowerSystem = PowerSystem("ice_ammonia_container_8000_teu")
	Tanks = [Tank("main_ammonia_container_8000_teu"), Tank("pilot_oil_container_8000_teu")]
	
	Route = Route("globalized_container_8000_teu")
	
	NominalCapacity = 8000

	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 85e6
	OPEX = 2.1e6
}