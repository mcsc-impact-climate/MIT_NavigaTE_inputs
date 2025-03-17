# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Route "globalized_container_15000_teu" {
	RouteType=REGIONAL_TRIP
	Ports = [Port("port_africa"), Port("port_americas"), Port("port_asia"), Port("port_europe"), Port("port_middle_east")]
	PortCalls = [10, 10, 10, 10, 10]

	TimeAtSea = 0.657 # 240 days a year
	#ConditionDistribution = [0.10, 0.45, 0.45]
	ConditionDistribution = [0.10, 0.65, 0.25]
	Speeds = [12, 18, 21]
	#Speeds = [12.3, 17.3, 21.3]
	CapacityUtilizations = [0.65, 0.65, 0.65]
}
