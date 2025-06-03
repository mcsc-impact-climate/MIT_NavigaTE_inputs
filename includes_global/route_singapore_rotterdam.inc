############################ Bulk carriers ###################################
Route "singapore_rotterdam_bulk_carrier_handy" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[11439.25, 11439.25]              # Based on voyage data from https://www.marinetraffic.com for cargo vessels traveling Singapore <--> Rotterdam between May 14-20, most vessels take the Cape of Good Hope route (~11,700) nm, but a minority take the Suez Canal route (~8,400 nm). The resulting average is 11439.25 +/- 1058.44.
    Speeds=[8.1999998, 8.1999998]                     # This is estimated based on operational data from https://www.marinetraffic.com between May 14-20 for vessels of this class and size that departed from Singapore to Rotterdam between 2023-01-01 and 2025-05-07, based on their IMO numbers.
    CapacityUtilizations=[0.50, 0.50]          # This is an estimate from NavigaTE defaults for this vessel class and size., would be good to get actual operational data for the route
    PortDurations=[0.7465277777777780, 1.89]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

#Route "singapore_rotterdam_bulk_carrier_handy" {
#    RouteType=REGIONAL_TRIP
#
#    Ports=[Port("port_singapore"), Port("port_rotterdam")]
#    PortCalls = [2, 2]
#    TimeAtSea = 0.698 # 255 days a year
#    ConditionDistribution = [0.5, 0.5]
#    #Speeds = [11, 14]
#    Speeds = [8.1999998, 8.1999998]
#    CapacityUtilizations = [0.5, 0.5]
#}

Route "singapore_rotterdam_bulk_carrier_panamax" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[11439.25, 11439.25]              # Based on voyage data from https://www.marinetraffic.com for cargo vessels traveling Singapore <--> Rotterdam between May 14-20, most vessels take the Cape of Good Hope route (~11,700) nm, but a minority take the Suez Canal route (~8,400 nm). The resulting average is 11439.25 +/- 1058.44.
    Speeds=[11.033333333333300, 11.033333333333300]                    # This is estimated based on operational data from https://www.marinetraffic.com between May 14-20 for vessels of this class and size that departed from Singapore to Rotterdam between 2023-01-01 and 2025-05-07, based on their IMO numbers.
    CapacityUtilizations=[0.50, 0.50]          # This is an estimate from NavigaTE defaults for this vessel class and size. Would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 1.89]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

#Route "singapore_rotterdam_bulk_carrier_panamax" {
#    RouteType=REGIONAL_TRIP
#
#    Ports=[Port("port_singapore"), Port("port_rotterdam")]
#    PortCalls = [2, 2]
#    TimeAtSea = 0.698 # 255 days a year
#    ConditionDistribution = [0.5, 0.5]
#    #Speeds = [11, 14]
#    Speeds = [11.033333333333300, 11.033333333333300]
#    CapacityUtilizations = [0.5, 0.5]
#}

Route "singapore_rotterdam_bulk_carrier_capesize" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[11439.25, 11439.25]              # Based on voyage data from https://www.marinetraffic.com for cargo vessels traveling Singapore <--> Rotterdam between May 14-20, most vessels take the Cape of Good Hope route (~11,700) nm, but a minority take the Suez Canal route (~8,400 nm). The resulting average is 11439.25 +/- 1058.44.
    Speeds=[8.033333466666670, 8.033333466666670]                     # This is estimated based on operational data from https://www.marinetraffic.com between May 14-20 for vessels of this class and size that departed from Singapore to Rotterdam between 2023-01-01 and 2025-05-07, based on their IMO numbers.
    CapacityUtilizations=[0.50, 0.50]          # This is an estimate from NavigaTE defaults for this vessel class and size. Would be good to get actual operational data for the route
    PortDurations=[1.8969907407407400, 1.89]          # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

#Route "singapore_rotterdam_bulk_carrier_capesize" {
#    RouteType=REGIONAL_TRIP
#
#    Ports=[Port("port_singapore"), Port("port_rotterdam")]
#    PortCalls = [2, 2]
#    TimeAtSea = 0.698 # 255 days a year
#    ConditionDistribution = [0.5, 0.5]
#    #Speeds = [11, 14]
#    Speeds = [8.033333466666670, 8.033333466666670]
#    CapacityUtilizations = [0.5, 0.5]
#}

############################# Container ships ###################################

Route "singapore_rotterdam_container_8000_teu" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[11439.25, 11439.25]              # Based on voyage data from https://www.marinetraffic.com for cargo vessels traveling Singapore <--> Rotterdam between May 14-20, most vessels take the Cape of Good Hope route (~11,700) nm, but a minority take the Suez Canal route (~8,400 nm). The resulting average is 11439.25 +/- 1058.44.
    Speeds=[14.961194035820900, 14.961194035820900]                     # This is estimated based on operational data from https://www.marinetraffic.com between May 14-20 for vessels of this class and size that departed from Singapore to Rotterdam between 2023-01-01 and 2025-05-07, based on their IMO numbers.
    CapacityUtilizations=[0.65, 0.65]          # This is an estimate from NavigaTE defaults for this vessel class and size. wWuld be good to get actual operational data for the route
    PortDurations=[1.2992848258706500, 1.89]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

#Route "singapore_rotterdam_container_8000_teu" {
#    RouteType=REGIONAL_TRIP
#
#    Ports=[Port("port_singapore"), Port("port_rotterdam")]
#    PortCalls = [2, 2]
#    TimeAtSea = 0.698 # 255 days a year
#    ConditionDistribution = [0.5, 0.5]
#    #Speeds = [11, 14]
#    Speeds = [14.961194035820900, 14.961194035820900]
#    CapacityUtilizations = [0.65, 0.65]
#}

Route "singapore_rotterdam_container_3500_teu" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[11439.25, 11439.25]              # Based on voyage data from https://www.marinetraffic.com for cargo vessels traveling Singapore <--> Rotterdam between May 14-20, most vessels take the Cape of Good Hope route (~11,700) nm, but a minority take the Suez Canal route (~8,400 nm). The resulting average is 11439.25 +/- 1058.44.
    Speeds=[13.25172412413790, 13.25172412413790]                     # This is estimated based on operational data from https://www.marinetraffic.com between May 14-20 for vessels of this class and size that departed from Singapore to Rotterdam between 2023-01-01 and 2025-05-07, based on their IMO numbers.
    CapacityUtilizations=[0.65, 0.65]          # This is an estimate from NavigaTE defaults for this vessel class and size. Would be good to get actual operational data for the route
    PortDurations=[1.6857040229885100, 1.89]          # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

#Route "singapore_rotterdam_container_3500_teu" {
#    RouteType=REGIONAL_TRIP
#
#    Ports=[Port("port_singapore"), Port("port_rotterdam")]
#    PortCalls = [2, 2]
#    TimeAtSea = 0.698 # 255 days a year
#    ConditionDistribution = [0.5, 0.5]
#    #Speeds = [11, 14]
#    Speeds = [13.25172412413790, 13.25172412413790]
#    CapacityUtilizations = [0.65, 0.65]
#}

############################ Tankers ###################################

Route "singapore_rotterdam_tanker_100k_dwt" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[11439.25, 11439.25]              # Based on voyage data from https://www.marinetraffic.com for cargo vessels traveling Singapore <--> Rotterdam between May 14-20, most vessels take the Cape of Good Hope route (~11,700) nm, but a minority take the Suez Canal route (~8,400 nm). The resulting average is 11439.25 +/- 1058.44.
    Speeds=[9.986666633333330, 9.986666633333330]                     # This is estimated based on operational data from https://www.marinetraffic.com between May 14-20 for vessels of this class and size that departed from Singapore to Rotterdam between 2023-01-01 and 2025-05-07, based on their IMO numbers.
    CapacityUtilizations=[0.60, 0.60]          # This is an estimate from NavigaTE defaults for this vessel class and size. Would be good to get actual operational data for the route
    PortDurations=[0.9300925925925930, 1.89]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

#Route "singapore_rotterdam_tanker_100k_dwt" {
#    RouteType=REGIONAL_TRIP
#
#    Ports=[Port("port_singapore"), Port("port_rotterdam")]
#    PortCalls = [2, 2]
#    TimeAtSea = 0.698 # 255 days a year
#    ConditionDistribution = [0.5, 0.5]
#    #Speeds = [11, 14]
#    Speeds = [9.986666633333330, 9.986666633333330]
#    CapacityUtilizations = [0.60, 0.60]
#}


Route "singapore_rotterdam_tanker_35k_dwt" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[11439.25, 11439.25]              # Based on voyage data from https://www.marinetraffic.com for cargo vessels traveling Singapore <--> Rotterdam between May 14-20, most vessels take the Cape of Good Hope route (~11,700) nm, but a minority take the Suez Canal route (~8,400 nm). The resulting average is 11439.25 +/- 1058.44.
    Speeds=[10.172413751724100, 10.172413751724100]                     # This is estimated based on operational data from https://www.marinetraffic.com between May 14-20 for vessels of this class and size that departed from Singapore to Rotterdam between 2023-01-01 and 2025-05-07, based on their IMO numbers.
    CapacityUtilizations=[0.60, 0.60]          # This is an estimate from NavigaTE defaults for this vessel class and size. Would be good to get actual operational data for the route
    PortDurations=[1.6909722222222200, 1.89]          # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

#Route "singapore_rotterdam_tanker_35k_dwt" {
#    RouteType=REGIONAL_TRIP
#
#    Ports=[Port("port_singapore"), Port("port_rotterdam")]
#    PortCalls = [2, 2]
#    TimeAtSea = 0.698 # 255 days a year
#    ConditionDistribution = [0.5, 0.5]
#    #Speeds = [11, 14]
#    Speeds = [10.172413751724100, 10.172413751724100]
#    CapacityUtilizations = [0.60, 0.60]
#}
