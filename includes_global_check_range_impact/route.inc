# DMM: Need to use a single port for the moment since we don't have emissions info for all ports.

Import Route "globalized_container_15000_teu"
Route "globalized_container_15000_teu" {
    Ports=[Port("Australia")]
    PortCalls=[40]
}

# Assign a route type
# ROUND_TRIP --> this is an ordered list, specifies routing
#       Also need to specify distances between ports
#       Will stop simulating if you don't have sufficient range
#

