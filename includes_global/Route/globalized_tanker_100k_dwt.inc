# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Route "globalized_tanker_100k_dwt" {
	RouteType=REGIONAL_TRIP
	
	Ports = [Port("port_africa"), Port("port_americas"), Port("port_asia"), Port("port_europe"), Port("port_middle_east")]
	PortCalls = [8, 8, 8, 8, 8]
	
	TimeAtSea = 0.616 #225 days at sea
	ConditionDistribution = [0.4, 0.6]
	#Speeds = [12, 13]
	#Speeds = [13.1, 14.1]
	Speeds = [12.5, 13.5]
	CapacityUtilizations = [0.0, 1.0]
}
