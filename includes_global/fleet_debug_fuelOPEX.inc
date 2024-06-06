# Definition of the fleet
Fleet "fleet_panamax" {
    Vessels = [Vessel("vessel_panamax_hydrogen")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 1
    TradeGrowth = 0
}

Fleet "fleet_capesize" {
    Vessels = [Vessel("vessel_capesize_hydrogen")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 1
    TradeGrowth = 0
}

# Define uptake node
Uptake "uptake" {
    Objective=MINIMIZE
}
