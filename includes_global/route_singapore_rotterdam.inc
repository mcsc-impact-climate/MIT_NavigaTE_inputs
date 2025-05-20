############################# Bulk carriers ###################################
Route "singapore_rotterdam_bulk_carrier_handy" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[11.35, 11.35]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.50, 0.50]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7465277777777780, 0.7465277777777780]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_bulk_carrier_panamax" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[12.55, 12.55]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.50, 0.50]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 0.7262731481481480]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_bulk_carrier_capesize" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[12.40, 12.40]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.50, 0.50]          # This is an estimate based on the vessel class. Would be good to get actual operational data for the route
    PortDurations=[1.8969907407407400, 1.8969907407407400]          # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

############################# Container ships ###################################
Route "singapore_rotterdam_container_15000_teu" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[18.15, 18.15]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.65, 0.65]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 0.7262731481481480]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_container_8000_teu" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[17.70, 17.70]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.65, 0.65]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 0.7262731481481480]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_container_3500_teu" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[16.05, 16.05]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.65, 0.65]          # This is an estimate based on the vessel class. Would be good to get actual operational data for the route
    PortDurations=[1.8969907407407400, 1.8969907407407400]          # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

############################# Tankers ###################################
Route "singapore_rotterdam_tanker_300k_dwt" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[13.48, 13.48]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.60, 0.60]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 0.7262731481481480]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_tanker_100k_dwt" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[13.10, 13.10]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.60, 0.60]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 0.7262731481481480]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_tanker_35k_dwt" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[10.00, 10.00]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.60, 0.60]          # This is an estimate based on the vessel class. Would be good to get actual operational data for the route
    PortDurations=[1.8969907407407400, 1.8969907407407400]          # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}
