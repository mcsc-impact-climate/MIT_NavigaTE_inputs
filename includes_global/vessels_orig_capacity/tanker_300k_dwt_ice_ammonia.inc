# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.


Vessel "tanker_300k_dwt_ice_ammonia" {
	
	PropulsionLoadLevel = Surface("speed_draft_power_surface_tanker_300k_dwt")
	ElectricalLoadLevelAtSea = 1
	ElectricalLoadLevelInPort = 1.2
	HeatLoadLevelAtSea = 4.0	
	HeatLoadLevelInPort = 4.0	
	
	PowerSystem = PowerSystem("ice_ammonia_tanker_300k_dwt")
	Tanks = [Tank("main_ammonia_tanker_300k_dwt"), Tank("pilot_oil_tanker_300k_dwt")]
	
	Route = Route("globalized_tanker_300k_dwt")
	
	NominalCapacity = 300000
	
	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 90e6
	OPEX = 2.9e6
}