# Definition of bulk fleet
Fleet "bulk_fleet" {
    Vessels = [Vessel("ammonia_vessel")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 100
    TradeGrowth = 0.05
}
