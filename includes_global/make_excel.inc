Report "excel_report" {
    Directory="./plots"
    
    # Vessel properties
    add_vessel_property("*", TotalEquivalentWTT)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentTTW)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentWTW)        # tonnes CO2e/year
    add_vessel_property("*", Miles)                     # miles/year
    add_vessel_property("*", CargoMiles)                # tonne/miles per year
    add_vessel_property("*", SpendEnergy, BOTH)         # Energy demand / year (GJ)
        
    add_vessel_property("*", TotalCAPEX)                
    add_vessel_property("*", TotalExcludingFuelOPEX)
    add_vessel_property("*", TotalFuelOPEX)
    add_vessel_property("*", ConsumedEnergy)            # Dictionary for each fuel
    
    # Fleet properties
    #add_fleet_property("*", Vessels)                   # DMM: Currently produces an error
    add_fleet_property("*", TotalEquivalentWTT)
    add_fleet_property("*", TotalEquivalentTTW)
    add_fleet_property("*", TotalEquivalentWTW)
    #add_fleet_property("*", Miles)
    #add_fleet_property("*", SpendEnergy, BOTH)         # DMM: Absolute number in GJ
    
    #add_fleet_property("*", TotalCAPEX)
    #add_fleet_property("*", TotalExcludingFuelOPEX)
    #add_fleet_property("*", TotalFuelOPEX)
    add_fleet_property("*", ConsumedEnergy)
}
