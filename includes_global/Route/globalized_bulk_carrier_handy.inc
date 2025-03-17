# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Route "globalized_bulk_carrier_handy" {
	RouteType=REGIONAL_TRIP
	Ports = [Port("port_africa"), Port("port_americas"), Port("port_asia"), Port("port_europe"), Port("port_middle_east")]
	PortCalls = [4, 4, 4, 4, 4]
	
	TimeAtSea = 0.575 # 210 days a year
	ConditionDistribution = [0.5, 0.5]
	#Speeds = [11.5, 12]
	Speeds = [11.1, 11.6]
	CapacityUtilizations = [1.0, 0.0]
}
