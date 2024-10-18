# Definition of fleets' uptake
Fleet "*" {                         # * applies this uptake to all defined nodes
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
}

Uptake "uptake" {
    Objective = MINIMIZE
}

