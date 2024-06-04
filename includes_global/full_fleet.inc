# Definition of the fleet
Fleet "fleet" {
    Vessels = [Vessel("vessel_baseline"), Vessel("vessel_alternative")]
    FuelUptake = Uptake("uptake")
    TechnologyUptake = Uptake("uptake")
    InitialVessels = 1000
    TradeGrowth = 0
    FixedScrapRate = 0.00        # Corresponds to 4% turnover per year, so 25 year lifetime
                                 # DMM: Set scrap rate to 0 to prevent fleet turnover
    InitialSplit = [0.5,0.5]     # DMM: Then the initial split set here stays constant over time
    
    InitialAges = Array [
        [0],
        [0]
    ]
    
    set_allow_vessel("vessel_baseline", FALSE)
}

# Define uptake node
Uptake "uptake" {
    Objective=MINIMIZE
}
