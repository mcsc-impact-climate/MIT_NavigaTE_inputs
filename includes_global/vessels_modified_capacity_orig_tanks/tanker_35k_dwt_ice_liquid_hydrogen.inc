Vessel "tanker_35k_dwt_ice_liquid_hydrogen" {
    
    PropulsionLoadLevel = Surface("speed_draft_power_surface_tanker_35k_dwt")
    ElectricalLoadLevelAtSea=0.6
    ElectricalLoadLevelInPort=1.2
    HeatLoadLevelAtSea=3.7
    HeatLoadLevelInPort=3.7
    
    PowerSystem=PowerSystem("ice_hydrogen_tanker_35k_dwt")   # DMM: This will need to change for hydrogen
    Tanks = [Tank("main_liquid_hydrogen_tanker_35k_dwt"), Tank("pilot_oil_tanker_35k_dwt")]
    
    Route=Route("globalized_tanker_35k_dwt")
    
    NominalCapacity = 27413.403237985378

    Lifetime=30
    CostOfCapital = Variable("cost_of_capital_vessel")
    CAPEX = 30e6
    OPEX = 2.3e6
    
}

PowerSystem "ice_hydrogen_tanker_35k_dwt" {
    Setup = INDIVIDUAL
    Propulsion = Converter("propulsion_ice_hydrogen_tanker_35k_dwt")
    Electrical = Converter("electrical_ice_hydrogen_tanker_35k_dwt")
    Heat = Converter("heat_boiler_oil_tanker_35k_dwt")

    CAPEX = 3.64e6
    OPEX = 3.64e4
}

Converter "propulsion_ice_hydrogen_tanker_35k_dwt" {
    PowerCapacity=8.5
    MinimumLoad=0.2
    MainFuelTypes = [HYDROGEN]
    PilotFuelTypes = [HFO]
    MinimumPilotFuel = 0.05
    Efficiency = Forecast("propulsion_ice_hydrogen_thermal_efficiency")
    
    set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
    set_consumption_TTW(HYDROGEN, "black_carbon", Variable("converter_hydrogen_black_carbon"))
    
    CAPEX = 700000
    OPEX = <700000 * 0.015>
}

Converter "electrical_ice_hydrogen_tanker_35k_dwt" {
    PowerCapacity=3.
    MainFuelTypes = [HYDROGEN]
    PilotFuelTypes = [HFO]
    MinimumPilotFuel = 0.05
    Efficiency = 0.48
    
    set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
    set_consumption_TTW(HYDROGEN, "black_carbon", Variable("converter_hydrogen_black_carbon"))
    
    CAPEX = 480000
    OPEX = 48000
}

# Description: Thermal efficiency of the propulsion (main engine) ICE using hydrogen
# Unit: GJ / GJ

# DMM: This may need to change for hydrogen (taken from ammonia)
Forecast "propulsion_ice_hydrogen_thermal_efficiency" {
    Table [
        "01-01-2024"    0.4925
        "01-01-2025"    0.4940
        "01-01-2026"    0.4954
        "01-01-2027"    0.4969
        "01-01-2028"    0.4984
        "01-01-2029"    0.4999
        "01-01-2030"    0.5014
        "01-01-2031"    0.5029
        "01-01-2032"    0.5044
        "01-01-2033"    0.5059
        "01-01-2034"    0.5075
        "01-01-2035"    0.5090
        "01-01-2036"    0.5105
        "01-01-2037"    0.5120
        "01-01-2038"    0.5136
        "01-01-2039"    0.5151
        "01-01-2040"    0.5167
        "01-01-2041"    0.5182
        "01-01-2042"    0.5198
        "01-01-2043"    0.5213
        "01-01-2044"    0.5229
        "01-01-2045"    0.5244
        "01-01-2046"    0.5260
        "01-01-2047"    0.5276
        "01-01-2048"    0.5292
        "01-01-2049"    0.5308
        "01-01-2050"    0.5324
    ]
    Extrapolate=LINEAR
}

