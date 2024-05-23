Report "excel_report" {
    Directory="./plots"
    add_vessel_property("*", TotalEquivalentWTT)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentTTW)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentWTW)        # tonnes CO2e/year
    add_vessel_property("*", Miles)
    add_vessel_property("*", TotalCAPEX)                #
    add_vessel_property("*", TotalExcludingFuelOPEX)
    add_vessel_property("*", TotalFuelOPEX)
    add_property(FeedstockMass)
    add_property(SourceEnergy)
    add_vessel_property("*", ConsumedEnergy)

}
