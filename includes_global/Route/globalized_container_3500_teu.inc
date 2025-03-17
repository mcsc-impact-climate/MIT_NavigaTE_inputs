# Copyright (c) 2024, Fonden Mærsk Mc-Kinney Møller Center for Zero Carbon Shipping.
#
# This file is part of NavigaTE.
#
# NavigaTE is a free software: you can modify it under the
# terms of the full license agreement found in the LICENSE
# file located in the root directory of this source tree.

Route "globalized_container_3500_teu" {
	RouteType=REGIONAL_TRIP
	Ports = [Port("port_africa"), Port("port_americas"), Port("port_asia"), Port("port_europe"), Port("port_middle_east")]
	PortCalls = [20, 20, 20, 20, 20]
	
	TimeAtSea = 0.575 # 210 days a year
	#ConditionDistribution = [0.15, 0.60, 0.25]
	ConditionDistribution = [0.20, 0.75, 0.05]
	Speeds = [12, 16.8, 21]
	#Speeds = [11.2, 16.2, 20.2]
	#Speeds = [11, 16, 20]
	CapacityUtilizations = [0.65, 0.65, 0.65]
}
