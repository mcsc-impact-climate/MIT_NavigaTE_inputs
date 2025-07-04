# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Vessel "bulk_carrier_panamax_ice_methanol" {
	
	PropulsionLoadLevel = Surface("speed_draft_power_surface_bulk_carrier_panamax")
	ElectricalLoadLevelAtSea = 0.34
	ElectricalLoadLevelInPort = 0.3
	HeatLoadLevelAtSea = 0	
	HeatLoadLevelInPort = 0.44
	
	PowerSystem = PowerSystem("ice_methanol_bulk_carrier_panamax")
	Tanks = [Tank("main_methanol_bulk_carrier_panamax"), Tank("pilot_oil_bulk_carrier_panamax")]
	
	Route = Route("globalized_bulk_carrier_panamax")
	
	NominalCapacity = 75000
	

	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 30e6
	OPEX = 1.8e6
}