# Definition of container fleet
Fleet "container_fleet" {
    Vessels = [Vessel("fuel_oil_vessel")]     # Adding methane vessel
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 100
    TradeGrowth = 0.05          # Annual growth of the fleet
}
