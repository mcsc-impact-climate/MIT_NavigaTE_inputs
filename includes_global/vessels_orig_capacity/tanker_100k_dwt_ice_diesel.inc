# DMM: Copied from tanker_100k_dwt_ice_oil.inc
Vessel "tanker_100k_dwt_ice_diesel" {
    
    PropulsionLoadLevel = Surface("speed_draft_power_surface_tanker_100k_dwt")
    ElectricalLoadLevelAtSea=0.75
    ElectricalLoadLevelInPort=1.2
    HeatLoadLevelAtSea=1.63
    HeatLoadLevelInPort=1.63
    
    PowerSystem=PowerSystem("ice_diesel_tanker_100k_dwt")
    Tanks = [Tank("main_diesel_tanker_100k_dwt")]
    
    Route=Route("globalized_tanker_100k_dwt")
    
    NominalCapacity = 100000

    Lifetime=25
    CostOfCapital = Variable("cost_of_capital_vessel")
    CAPEX = 50e6
    OPEX = 2.5e6
    
}

PowerSystem "ice_diesel_tanker_100k_dwt" {
    Propulsion = [Converter("propulsion_ice_diesel_tanker_100k_dwt")]
    Electrical = [Converter("electrical_ice_diesel_tanker_100k_dwt")]
    Heat = [Converter("heat_boiler_oil_diesel_tanker_100k_dwt")]
    
    PropulsionInStreams = Array [
        [0, 1.0, 0, 0]
    ]
    
    ElectricalInStreams = Array [
        [0, 0, 1.0, 0]
    ]
    
    HeatInStreams = Array [
        [0, 0, 0, 0.825]
    ]
    
    CAPEX = 0.325e6     # DMM: Copied from ice_oil_tanker_100k_dwt.inc
    OPEX = 0.325e4      # DMM: Copied from ice_oil_tanker_100k_dwt.inc
}

Converter "propulsion_ice_diesel_tanker_100k_dwt" {
    PowerCapacity=13.9
    MinimumLoad=0.2
    MainFuelTypes = [DIESEL]
    OutStream = [0, Forecast("propulsion_ice_diesel_thermal_efficiency"), 0, <1 - Forecast("propulsion_ice_diesel_thermal_efficiency")>]
    PrimaryStream = MECHANICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 520000               # DMM: Copied from propulsion_ice_oil_tanker_100k_dwt.inc
    OPEX = <520000 * 0.01>      # DMM: Copied from propulsion_ice_oil_tanker_100k_dwt.inc
}

Converter "electrical_ice_diesel_tanker_100k_dwt" {
    PowerCapacity=3.
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0.48, 0.52]
    PrimaryStream = ELECTRICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 400000      # DMM: Copied from electrical_ice_oil_tanker_100k_dwt.inc
    OPEX = 40000        # DMM: Copied from electrical_ice_oil_tanker_100k_dwt.inc
}

# Description: Thermal efficiency of the propulsion (main engine) ICE using diesel
# Unit: GJ / GJ

# DMM: This may need to change for diesel (taken from ammonia)
Forecast "propulsion_ice_diesel_thermal_efficiency" {
    Table [
        "01-01-2024"    0.5143
        "01-01-2025"    0.5158
        "01-01-2026"    0.5174
        "01-01-2027"    0.5189
        "01-01-2028"    0.5205
        "01-01-2029"    0.5220
        "01-01-2030"    0.5236
        "01-01-2031"    0.5252
        "01-01-2032"    0.5268
        "01-01-2033"    0.5283
        "01-01-2034"    0.5299
        "01-01-2035"    0.5315
        "01-01-2036"    0.5331
        "01-01-2037"    0.5347
        "01-01-2038"    0.5363
        "01-01-2039"    0.5379
        "01-01-2040"    0.5395
        "01-01-2041"    0.5411
        "01-01-2042"    0.5428
        "01-01-2043"    0.5444
        "01-01-2044"    0.5460
        "01-01-2045"    0.5477
        "01-01-2046"    0.5493
        "01-01-2047"    0.5510
        "01-01-2048"    0.5526
        "01-01-2049"    0.5543
        "01-01-2050"    0.5559
    ]
    Extrapolate=LINEAR
}

Converter "heat_boiler_oil_diesel_tanker_100k_dwt" {
    PowerCapacity=2.
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0, 1.0]
    PrimaryStream = HEAT
    
    CAPEX = 55e3
    OPEX = 0
}

