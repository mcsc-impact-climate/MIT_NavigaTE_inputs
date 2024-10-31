# Definition of the fleet
Fleet "fleet" {
    Vessels = [Vessel("vessel")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 1
    TradeGrowth = 0
}

# Define uptake node
Uptake "uptake" {
    Objective=MINIMIZE
}
