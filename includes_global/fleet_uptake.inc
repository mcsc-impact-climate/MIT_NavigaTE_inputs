# Definition of fleets' uptake
Fleet "*" {                         # * applies this uptake to all defined nodes
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
}

Uptake "uptake" {
    Method = MULTINOMIAL
    Objective = MINIMIZE
    Beta = 1
    CoefficientOfVariation = 0.15
}

