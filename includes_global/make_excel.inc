Report "excel_report" {
    Directory="./plots"
    add_vessel_property("*", TotalEquivalentWTT)
    add_vessel_property("*", TotalEquivalentTTW)
    add_vessel_property("*", TotalEquivalentWTW)
    add_vessel_property("*", Miles)
    add_property(FeedstockMass)
    add_property(SourceEnergy)
    add_vessel_property("*", ConsumedEnergy)

}
