# DMM: Copied from bulk_carrier_handy_ice_oil.inc
Vessel "bulk_carrier_handy_ice_FTdiesel" {
    
    PropulsionLoadLevel = Surface("speed_draft_power_surface_bulk_carrier_handy")
    ElectricalLoadLevelAtSea=0.3
    ElectricalLoadLevelInPort=0.5
    HeatLoadLevelAtSea=0.25
    HeatLoadLevelInPort=0.49
    
    PowerSystem=PowerSystem("ice_diesel_bulk_carrier_handy")
    Tanks = [Tank("main_diesel_bulk_carrier_handy")]
    
    Route=Route("globalized_bulk_carrier_handy")
    
    NominalCapacity = 21419.991691055973

    Lifetime=25
    Finance=Finance("vessel_finance")
    CAPEX = 23e6
    OPEX = 1.5e6
    
}

PowerSystem "ice_diesel_bulk_carrier_handy" {
    Propulsion = [Converter("propulsion_ice_diesel_bulk_carrier_handy")]
    Electrical = [Converter("electrical_ice_diesel_bulk_carrier_handy")]
    Heat = [Converter("heat_boiler_oil_diesel_bulk_carrier_handy")]
    
    PropulsionInStreams = Array [
        [0, 1.0, 0, 0]
    ]
    
    ElectricalInStreams = Array [
        [0, 0, 1.0, 0]
    ]
    
    HeatInStreams = Array [
        [0, 0, 0, 0.825]
    ]
    
    CAPEX = 0.22e6      # DMM: Copied from ice_oil_bulk_carrier_handy.inc
    OPEX = 0.22e4       # DMM: Copied from ice_oil_bulk_carrier_handy.inc
}

Converter "propulsion_ice_diesel_bulk_carrier_handy" {
    PowerCapacity=7.1       # DMM: Copied from propulsion_ice_oil_bulk_carrier_handy.inc
    MinimumLoad=0.2         # DMM: Copied from propulsion_ice_oil_bulk_carrier_handy.inc
    MainFuelTypes = [DIESEL]
    OutStream = [0, Forecast("propulsion_ice_diesel_thermal_efficiency"), 0, <1 - Forecast("propulsion_ice_diesel_thermal_efficiency")>]
    PrimaryStream = MECHANICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 520000              # DMM: Copied from propulsion_ice_oil_bulk_carrier_handy.inc
    OPEX = <520000 * 0.01>      # DMM: Copied from propulsion_ice_oil_bulk_carrier_handy.inc
}

Converter "electrical_ice_diesel_bulk_carrier_handy" {
    PowerCapacity=1.7
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0.5, 0.5]        # DMM: Copied from electrical_ice_oil_bulk_carrier_handy.inc
    PrimaryStream = ELECTRICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 400000
    OPEX = 40000
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

Converter "heat_boiler_oil_diesel_bulk_carrier_handy" {
    PowerCapacity=2.
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0, 1.0]
    PrimaryStream = HEAT
    
    CAPEX = 55e3
    OPEX = 0
}

