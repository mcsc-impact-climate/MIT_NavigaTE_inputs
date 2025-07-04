# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Vessel "bulk_carrier_capesize_ice_ammonia" {
	
	PropulsionLoadLevel = Surface("speed_draft_power_surface_bulk_carrier_capesize")
	ElectricalLoadLevelAtSea = 0.7
	ElectricalLoadLevelInPort = 0.7
	HeatLoadLevelAtSea = 0.
	HeatLoadLevelInPort = 0.9
	
	PowerSystem = PowerSystem("ice_ammonia_bulk_carrier_capesize")
	Tanks = [Tank("main_ammonia_bulk_carrier_capesize"), Tank("pilot_oil_bulk_carrier_capesize")]
	
	Route = Route("globalized_bulk_carrier_capesize")
	
	NominalCapacity = 200000
	

	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 50e6
	OPEX = 2.4e6
}