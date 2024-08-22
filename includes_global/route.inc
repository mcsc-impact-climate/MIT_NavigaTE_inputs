# DMM: Need to use a single port for the moment since we don't have emissions info for all ports.

Import Route "globalized_bulk_carrier_panamax"
Route "globalized_bulk_carrier_panamax" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_bulk_carrier_capesize"
Route "globalized_bulk_carrier_capesize" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_bulk_carrier_handy"
Route "globalized_bulk_carrier_handy" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_tanker_35k_dwt"
Route "globalized_tanker_35k_dwt" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_tanker_300k_dwt"
Route "globalized_tanker_300k_dwt" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_tanker_100k_dwt"
Route "globalized_tanker_100k_dwt" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_container_8000_teu"
Route "globalized_container_8000_teu" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_container_3500_teu"
Route "globalized_container_3500_teu" {
    Ports=[Port("port")]
    PortCalls=[40]
}

Import Route "globalized_container_15000_teu"
Route "globalized_container_15000_teu" {
    Ports=[Port("port")]
    PortCalls=[40]
}

# Assign a route type
# ROUND_TRIP --> this is an ordered list, specifies routing
#       Also need to specify distances between ports
#       Will stop simulating if you don't have sufficient range
#
Import Route "globalized_gas_carrier_100k_cbm"
Route "globalized_gas_carrier_100k_cbm" {
    Ports=[Port("port")]
    PortCalls=[40]
}

