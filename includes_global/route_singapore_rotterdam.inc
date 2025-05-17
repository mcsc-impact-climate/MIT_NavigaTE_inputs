Route "singapore_rotterdam_bulk_carrier_handy" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[11.28205, 11.28205]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.61000, 0.61000]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7465277777777780, 0.7465277777777780]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_bulk_carrier_panamax" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[11.36430, 11.36430]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.57990, 0.57990]          # This is an estimate, would be good to get actual operational data for the route
    PortDurations=[0.7262731481481480, 0.7262731481481480]        # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}

Route "singapore_rotterdam_bulk_carrier_capesize" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("port_singapore"), Port("port_rotterdam")]
    Distances=[8419, 8419]              # Would be good to get distribution
    Speeds=[11.18760, 11.18760]                     # This is an estimate based on the vessel class. Would be good to get actual operational data for this route
    CapacityUtilizations=[0.57010, 0.57010]          # This is an estimate based on the vessel class. Would be good to get actual operational data for the route
    PortDurations=[1.8969907407407400, 1.8969907407407400]          # Singapore port duration is based on the Singapore port data. Rotterdam port duration is currently copied from Singpore, but should be updated with Rotterdam-specific data.
}
