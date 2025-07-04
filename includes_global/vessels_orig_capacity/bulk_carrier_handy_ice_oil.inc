# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Vessel "bulk_carrier_handy_ice_oil" {

	PropulsionLoadLevel = Surface("speed_draft_power_surface_bulk_carrier_handy")
	ElectricalLoadLevelAtSea = 0.3
	ElectricalLoadLevelInPort = 0.5
	HeatLoadLevelAtSea = 0.25
	HeatLoadLevelInPort = 0.49
	
	PowerSystem = PowerSystem("ice_oil_bulk_carrier_handy")
	Tanks = [Tank("main_oil_bulk_carrier_handy")]
	
	Route = Route("globalized_bulk_carrier_handy")
	
	NominalCapacity = 35000
	

	Lifetime = 25
	CostOfCapital = Variable("cost_of_capital_vessel")
	CAPEX = 23e6
	OPEX = 1.5e6
}