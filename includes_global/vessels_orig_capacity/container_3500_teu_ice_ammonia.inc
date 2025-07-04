# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Vessel "container_3500_teu_ice_ammonia" {
	
	PropulsionLoadLevel = Curve("speed_power_curve_container_3500_teu")
	ElectricalLoadLevelAtSea = 0.9
	ElectricalLoadLevelInPort = 0.7
	HeatLoadLevelAtSea = 0.3
	HeatLoadLevelInPort = 0.74
	
	PowerSystem = PowerSystem("ice_ammonia_container_3500_teu")
	Tanks = [Tank("main_ammonia_container_3500_teu"), Tank("pilot_oil_container_3500_teu")]
	
	Route = Route("globalized_container_3500_teu")
	
	NominalCapacity = 3500

	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 40e6
	OPEX = 1.8e6
}