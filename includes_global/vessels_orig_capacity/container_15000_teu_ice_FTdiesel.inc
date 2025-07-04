Vessel "container_15000_teu_ice_FTdiesel" {
    
    PropulsionLoadLevel = Curve("speed_power_curve_container_15000_teu")
    ElectricalLoadLevelAtSea=1.4
    ElectricalLoadLevelInPort=1.3
    HeatLoadLevelAtSea=0.15
    HeatLoadLevelInPort=1.48
    
    PowerSystem=PowerSystem("ice_diesel_container_15000_teu")
    Tanks = [Tank("main_FTdiesel_container_15000_teu")]
    
    Route=Route("globalized_container_15000_teu")
    
    NominalCapacity = 15000

    Lifetime=25
    CostOfCapital = Variable("cost_of_capital_vessel")
    CAPEX = 140e6
    OPEX = 2.5e6
    
}

PowerSystem "ice_diesel_container_15000_teu" {
    Setup = INDIVIDUAL
    Propulsion = Converter("propulsion_ice_diesel_container_15000_teu")
    Electrical = Converter("electrical_ice_diesel_container_15000_teu")
    Heat = Converter("heat_boiler_diesel_container_15000_teu")
    
    CAPEX = 1.22e6      # DMM: Copied from ice_oil_container_15000_teu
    OPEX = 1.22e4       # DMM: Copied from ice_oil_container_15000_teu
}

Converter "propulsion_ice_diesel_container_15000_teu" {
    PowerCapacity=68.6      # DMM: Copied from propulsion_ice_oil_container_15000_teu.inc
    MinimumLoad=0.15        # DMM: Copied from propulsion_ice_oil_container_15000_teu.inc
    MainFuelTypes = [DIESEL]
    Efficiency = Forecast("propulsion_ice_diesel_thermal_efficiency")
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 520000                  # DMM: Copied from propulsion_ice_oil_container_15000_teu.inc
    OPEX = <520000 * 0.01>          # DMM: Copied from propulsion_ice_oil_container_15000_teu.inc
}

Converter "electrical_ice_diesel_container_15000_teu" {
    PowerCapacity=14.5
    MainFuelTypes = [DIESEL]
    Efficiency = 0.48
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 400000             # DMM: Copied from electrical_ice_oil_container_15000_teu.inc
    OPEX = 40000               # DMM: Copied from electrical_ice_oil_container_15000_teu.inc
}

# Description: Thermal efficiency of the propulsion (main engine) ICE using diesel
# Unit: GJ / GJ

# DMM: This is copied from oil, may need to change for diesel
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

