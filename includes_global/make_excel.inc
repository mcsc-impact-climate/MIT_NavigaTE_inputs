Report "excel_report" {
    Directory="../../all_outputs_full_fleet"
    
    # Vessel properties
    add_vessel_property("*", TotalEquivalentWTT)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentTTW)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentWTW)        # tonnes CO2e/year
    add_vessel_property("*", Miles)                     # miles/year
    add_vessel_property("*", CargoMiles)                # tonne/miles per year
    add_vessel_property("*", SpendEnergy, BOTH)         # Energy demand / year (GJ)
        
    add_vessel_property("*", CAPEX)
    add_vessel_property("*", OPEX)
    add_vessel_property("*", BaseCAPEX)
    add_vessel_property("*", BaseOPEX)
    add_vessel_property("*", TankCAPEX)
    add_vessel_property("*", TankOPEX)
    add_vessel_property("*", PowerCAPEX)
    add_vessel_property("*", PowerOPEX)
    add_vessel_property("*", FuelOPEX)
    add_vessel_property("*", TotalCost)
    add_vessel_property("*", ConsumedEnergy)            # Dictionary for each fuel
    
    # Fleet properties
    add_fleet_property("*", TotalEquivalentWTT)
    add_fleet_property("*", TotalEquivalentTTW)
    add_fleet_property("*", TotalEquivalentWTW)
    
    add_fleet_property("*", ConsumedEnergy)
}
