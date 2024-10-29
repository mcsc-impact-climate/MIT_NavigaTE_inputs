# DMM: Currently copied from container_3500_teu_ice_oil.inc
Vessel "container_3500_teu_ice_FTdiesel" {
    
    PropulsionLoadLevel = Curve("speed_power_curve_container_3500_teu")
    ElectricalLoadLevelAtSea=0.9
    ElectricalLoadLevelInPort=0.7
    HeatLoadLevelAtSea=0.3
    HeatLoadLevelInPort=0.74
    
    PowerSystem=PowerSystem("ice_diesel_container_3500_teu")
    Tanks = [Tank("main_FTdiesel_container_3500_teu"), Tank("pilot_oil_bulk_carrier_capesize")]
    
    Route=Route("globalized_container_3500_teu")
    
    NominalCapacity = 3493.1988532800383

    Lifetime=25
    Finance=Finance("vessel_finance")
    CAPEX = 40e6
    OPEX = 1.8e6
    
    EnforceMainFuelShare = TRUE
}

PowerSystem "ice_diesel_container_3500_teu" {
    Propulsion = [Converter("propulsion_ice_diesel_container_3500_teu")]
    Electrical = [Converter("electrical_ice_diesel_container_3500_teu")]
    Heat = [Converter("heat_boiler_oil_diesel_container_3500_teu")]
    
    PropulsionInStreams = Array [
        [0, 1.0, 0, 0]
    ]
    
    ElectricalInStreams = Array [
        [0, 0, 1.0, 0]
    ]
    
    HeatInStreams = Array [
        [0, 0, 0, 0.825]
    ]
    
    CAPEX = 0.46e6      # DMM: Copied from ice_oil_container_3500_teu.inc
    OPEX = 0.46e4       # DMM: Copied from ice_oil_container_3500_teu.inc
}

Converter "propulsion_ice_diesel_container_3500_teu" {
    PowerCapacity=26.9              # DMM: Copied from propulsion_ice_oil_container_3500_teu.inc
    MinimumLoad=0.15                # DMM: Copied from propulsion_ice_oil_container_3500_teu.inc
    MainFuelTypes = [DIESEL]
    OutStream = [0, Forecast("propulsion_ice_diesel_thermal_efficiency"), 0, <1 - Forecast("propulsion_ice_diesel_thermal_efficiency")>]
    PrimaryStream = MECHANICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 520000              # DMM: Copied from propulsion_ice_oil_container_3500_teu.inc
    OPEX = <520000 * 0.01>      # DMM: Copied from propulsion_ice_oil_container_3500_teu.inc
}

Converter "electrical_ice_diesel_container_3500_teu" {
    PowerCapacity=6.
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0.5, 0.5]        # Copied from electrical_ice_oil_container_3500_teu.inc
    PrimaryStream = ELECTRICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 400000      # Copied from electrical_ice_oil_container_3500_teu.inc
    OPEX = 40000        # Copied from electrical_ice_oil_container_3500_teu.inc
}

# Description: Thermal efficiency of the propulsion (main engine) ICE using diesel
# Unit: GJ / GJ

# DMM: This may need to change for diesel (taken from oil)
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

Converter "heat_boiler_oil_diesel_container_3500_teu" {
    PowerCapacity=1.5       # DMM: Copied from heat_boiler_oil_container_3500_teu.inc
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0, 1.0]  # DMM: Copied from heat_boiler_oil_container_3500_teu.inc
    PrimaryStream = HEAT
    
    CAPEX = 55e3            # DMM: Copied from heat_boiler_oil_container_3500_teu.inc
    OPEX = 0                # DMM: Copied from heat_boiler_oil_container_3500_teu.inc
}

