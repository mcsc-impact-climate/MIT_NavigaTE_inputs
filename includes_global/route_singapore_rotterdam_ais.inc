############################ Bulk carriers ###################################
Route "singapore_rotterdam_bulk_carrier_handy" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[11.35, 11.35]                           # Weighted average over NavigaTE speeds
    CapacityUtilizations=[0.50, 0.50]               # This is an estimate from NavigaTE defaults for this vessel class and size., would be good to get actual operational data for the route
    PortDurations=[0.7465277777777780, 1.89]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

Route "singapore_rotterdam_bulk_carrier_panamax" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[12.55, 12.55]                           # Weighted average over NavigaTE speeds
    CapacityUtilizations=[0.50, 0.50]               # This is an estimate from NavigaTE defaults for this vessel class and size. Would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 1.89]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

Route "singapore_rotterdam_bulk_carrier_capesize" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[12.4, 12.4]                             # Weighted average over NavigaTE speeds
    CapacityUtilizations=[0.50, 0.50]               # This is an estimate from NavigaTE defaults for this vessel class and size. Would be good to get actual operational data for the route
    PortDurations=[1.8969907407407400, 1.89]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

############################# Container ships ###################################

Route "singapore_rotterdam_container_15000_teu" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[15.91, 15.85]                           # From Ioannis' AIS data analysis
    CapacityUtilizations=[0.730, 0.445]             # From Ioannis' AIS data analysis
    PortDurations=[0.991, 1.441]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

Route "singapore_rotterdam_container_8000_teu" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[16.12, 15.60]                           # From Ioannis' AIS data analysis
    CapacityUtilizations=[0.671, 0.616]             # From Ioannis' AIS data analysis
    PortDurations=[0.704, 0.993]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

Route "singapore_rotterdam_container_3500_teu" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[14.33, 16.54]                           # From Ioannis' AIS data analysis
    CapacityUtilizations=[0.716, 0.367]             # From Ioannis' AIS data analysis
    PortDurations=[0.147, 0.782]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

############################ Tankers ###################################

Route "singapore_rotterdam_tanker_300k_dwt" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[12.64, 12.40]                           # From Ioannis' AIS data analysis
    CapacityUtilizations=[0.505, 0.353]             # From Ioannis' AIS data analysis
    PortDurations=[0.995, 1.331]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

Route "singapore_rotterdam_tanker_100k_dwt" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[11.76, 11.83]                           # From Ioannis' AIS data analysis
    CapacityUtilizations=[0.547, 0.667]             # From Ioannis' AIS data analysis
    PortDurations=[0.468, 1.487]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

Route "singapore_rotterdam_tanker_35k_dwt" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[12.33, 12.27]                           # Weighted average over NavigaTE speeds
    CapacityUtilizations=[0.567, 0.482]             # From Ioannis' AIS data analysis
    PortDurations=[1.598, 2.523]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}

######################## Gas Carrier ###############################

Route "singapore_rotterdam_gas_carrier_100k_cbm" {
    RouteType=ROUND_TRIP

    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8440, 8440]                          # From Ioannis' AIS data analysis
    Speeds=[13.41, 13.41]                           # From Ioannis' AIS data analysis
    CapacityUtilizations=[0.596, 0.596]             # From Ioannis' AIS data analysis
    PortDurations=[1.095, 1.095]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is obtained by averaging the Time in Port under "Intra-port efficiency" in https://www.portofrotterdam.com/en/logistics/port-performance
}