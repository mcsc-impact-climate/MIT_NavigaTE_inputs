Report "excel_report" {
    Directory="./plots"
    add_vessel_property("*", TotalEquivalentWTT)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentTTW)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentWTW)        # tonnes CO2e/year
    add_vessel_property("*", Miles)                     # miles/year
    add_vessel_property("*", CargoMiles)                # tonne/miles per year
    #add_vessel_property("*", BunkerMass)                # tonnes/year
    add_vessel_property("*", SpendEnergy, FIRST)        # Energy demand / year (GJ)
        
    add_vessel_property("*", TotalCAPEX)                
    add_vessel_property("*", TotalExcludingFuelOPEX)
    add_vessel_property("*", TotalFuelOPEX)
    add_property(FeedstockMass)
    add_property(SourceEnergy)
    add_vessel_property("*", ConsumedEnergy)

}
