# Definition of vessel route 
Route "vessel_route" {
    RouteType = REGIONAL_TRIP
    Ports = [Port("route_port")]
    TimeAtSea = 0.75
    ConditionDistribution = [1.0]
    Speeds = [10]
}
