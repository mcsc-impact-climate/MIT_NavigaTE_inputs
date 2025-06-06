Report "excel_report" {
    Directory="./plots"
    
    # Vessel properties
    add_vessel_property("*", TotalEquivalentWTT)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentTTW)        # tonnes CO2e/year
    add_vessel_property("*", TotalEquivalentWTW)        # tonnes CO2e/year
    add_vessel_property("*", IntensityTotalEquivalentWTT)        # kg CO2e / GJ fuel
    add_vessel_property("*", IntensityTotalEquivalentTTW)        # kg CO2e / GJ fuel
    add_vessel_property("*", IntensityTotalEquivalentWTW)        # kg CO2e / GJ fuel
    add_vessel_property("*", Miles)                     # miles/year
    add_vessel_property("*", CargoMiles)                # tonne/miles per year
    add_vessel_property("*", SpendEnergy, BOTH)         # Energy demand / year (GJ)
        
    add_vessel_property("*", TotalCost)
    add_vessel_property("*", AssetCost)
    add_vessel_property("*", CAPEX)
    add_vessel_property("*", OPEX)
    add_vessel_property("*", BaseCAPEX)
    add_vessel_property("*", BaseOPEX)
    add_vessel_property("*", TankCAPEX)
    add_vessel_property("*", TankOPEX)
    add_vessel_property("*", PowerCAPEX)
    add_vessel_property("*", PowerOPEX)
    add_vessel_property("*", FuelOPEX)
    add_vessel_property("*", RegulationOPEX)
    add_vessel_property("*", InvestmentMetric)
    add_vessel_property("*", InvestmentMetricExpected)
    add_vessel_property("*", ConsumedEnergy)            # Dictionary for each fuel
    add_vessel_property("*", ConsumedVolume)            # Dictionary for each fuel
    add_vessel_property("*", PilotFuelShare)
    
    # Fleet properties
    add_fleet_property("*", ExistingVessels)
    add_fleet_property("*", Scrap)
    add_fleet_property("*", Newbuilds)
    add_fleet_property("*", FuelConversions)
    add_fleet_property("*", TotalEquivalentWTT)
    add_fleet_property("*", TotalEquivalentTTW)
    add_fleet_property("*", TotalEquivalentWTW)
    add_fleet_property("*", IntensityTotalEquivalentWTT)
    add_fleet_property("*", IntensityTotalEquivalentTTW)
    add_fleet_property("*", IntensityTotalEquivalentWTW)
    #add_fleet_property("*", Miles)
    #add_fleet_property("*", SpendEnergy, BOTH)         # DMM: Absolute number in GJ
    add_fleet_property("*", ConsumedEnergy)
    add_fleet_property("*", Trade)
    
    add_property(TotalEquivalentWTT)
    add_property(TotalEquivalentTTW)
    add_property(TotalEquivalentWTW)
    add_property(CumulativeTotalEquivalentWTW)
    add_property(IntensityTotalEquivalentWTT)
    add_property(IntensityTotalEquivalentTTW)
    add_property(IntensityTotalEquivalentWTW)
    add_property(ConsumedEnergy)
    add_property(FuelExpenses)
    add_property(RegulationExpenses)
    add_property(FuelRelatedExpenses)
    add_property(VesselExpenses)
    add_property(FuelConversionExpenses)
    add_property(VesselRelatedExpenses)
    add_property(Expenses)
    add_property(CumulativeExpenses)
    
    #add_property(StaticSupplyMass)
    #add_property(StaticSupplyEnergy)
    #add_property(DynamicSupplyMass)
    #add_property(DynamicSupplyEnergy)
    
#    add_property(ConverterFuelTypeEnergy)
    
#    add_property(FuelTypeDemand)
    # add_property(FuelTypeSupply)
        
#    add_property(CumulativeConsumedMass)
#    add_property(CumulativeFuelRelatedExpenses)
#    add_property(CumulativeRegulationExpenses)
        
#    add_property(CumulativeWTT)
#    add_property(CumulativeEquivalentTTW)
#    add_property(CumulativeTotalEquivalentWTW)
    
#    add_property(CumulativeIntensityWTT)
#    add_property(CumulativeIntensityEquivalentTTW)
#    add_property(CumulativeIntensityTotalEquivalentWTW)

    add_regulation_property("*", FlexibilityCost)
    add_regulation_property("*", RemedialCost)
    add_regulation_property("*", IntendedUnits)
    add_regulation_property("*", SurplusUnits)
    add_regulation_property("*", FlexibilityUnits)
    add_regulation_property("*", RemedialUnits)
    add_regulation_property("*", SurplusRevenue)
    add_regulation_property("*", FlexibilityExpenses)
    add_regulation_property("*", RemedialExpenses)
        
    add_regulation_property("*", VesselThreshold)
    
    add_port_property("*", BunkerMass)
    add_port_property("*", BunkerEnergy)
    add_port_property("*", BunkerPrice)
}
