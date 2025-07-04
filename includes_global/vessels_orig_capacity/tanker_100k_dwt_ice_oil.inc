# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.


Vessel "tanker_100k_dwt_ice_oil" {

	PropulsionLoadLevel = Surface("speed_draft_power_surface_tanker_100k_dwt")
	ElectricalLoadLevelAtSea = 0.75
	ElectricalLoadLevelInPort = 1.2
	HeatLoadLevelAtSea = 3.7
	HeatLoadLevelInPort = 3.7
	
	PowerSystem = PowerSystem("ice_oil_tanker_100k_dwt")
	Tanks = [Tank("main_oil_tanker_100k_dwt")]
	
	Route = Route("globalized_tanker_100k_dwt")
	
	NominalCapacity = 100000
	
	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 50e6
	OPEX = 2.5e6
}