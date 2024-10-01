# Definition of the hydrogen container fleet
Fleet "hydrogen_fleet_container" {
    Vessels = [Vessel("container_15000_teu_ice_hydrogen")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 100
    InitialSplit = [1] 
    TradeGrowth = 0
}

# Define uptake node
Uptake "uptake" {
    Objective=MINIMIZE
}
