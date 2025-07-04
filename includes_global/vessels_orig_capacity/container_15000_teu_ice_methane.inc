# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Vessel "container_15000_teu_ice_methane" {
	
	PropulsionLoadLevel = Curve("speed_power_curve_container_15000_teu")
	ElectricalLoadLevelAtSea = 1.4
	ElectricalLoadLevelInPort = 1.3
	HeatLoadLevelAtSea = 0.15
	HeatLoadLevelInPort = 1.48
	
	PowerSystem = PowerSystem("ice_methane_container_15000_teu")
	Tanks = [Tank("main_methane_container_15000_teu"), Tank("pilot_oil_container_15000_teu")]
	
	Route = Route("globalized_container_15000_teu")
	
	NominalCapacity = 15000

	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 140e6
	OPEX = 2.5e6
}