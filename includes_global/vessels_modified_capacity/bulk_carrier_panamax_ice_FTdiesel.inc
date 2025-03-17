# DMM: Copied from bulk_carrier_panamax_ice_oil.inc
Vessel "bulk_carrier_panamax_ice_FTdiesel" {
    
    PropulsionLoadLevel = Surface("speed_draft_power_surface_bulk_carrier_panamax")
    ElectricalLoadLevelAtSea=0.34
    ElectricalLoadLevelInPort=0.3       # Units of MW
    HeatLoadLevelAtSea=0
    HeatLoadLevelInPort=0.44
    
    PowerSystem=PowerSystem("ice_diesel_bulk_carrier_panamax")
    Tanks = [Tank("main_FTdiesel_bulk_carrier_panamax"), Tank("pilot_oil_bulk_carrier_capesize")]
    
    Route=Route("globalized_bulk_carrier_panamax")
    
    NominalCapacity = 75254.63194444444

    Lifetime=25
    Finance=Finance("vessel_finance")
    CAPEX = 30e6
    OPEX = 1.8e6
    
}

PowerSystem "ice_diesel_bulk_carrier_panamax" {
    Propulsion = [Converter("propulsion_ice_diesel_bulk_carrier_panamax")]
    Electrical = [Converter("electrical_ice_diesel_bulk_carrier_panamax")]
    Heat = [Converter("heat_boiler_oil_diesel_bulk_carrier_panamax")]
    
    PropulsionInStreams = Array [
        [0, 1.0, 0, 0]
    ]
    
    ElectricalInStreams = Array [
        [0, 0, 1.0, 0]
    ]
    
    HeatInStreams = Array [
        [0, 0, 0, 0.825]
    ]
    
    CAPEX = 0.253e6     # DMM: Taken from ice_oil_bulk_carrier_panamax.inc
    OPEX = 0.253e4      # DMM: Taken from ice_oil_bulk_carrier_panamax.inc
}

Converter "propulsion_ice_diesel_bulk_carrier_panamax" {
    PowerCapacity=10.3      # DMM: Copied from propulsion_ice_oil_bulk_carrier_panamax.inc
    MinimumLoad=0.2         # DMM: Copied from propulsion_ice_oil_bulk_carrier_panamax.inc
    MainFuelTypes = [DIESEL]
    OutStream = [0, Forecast("propulsion_ice_diesel_thermal_efficiency"), 0, <1 - Forecast("propulsion_ice_diesel_thermal_efficiency")>]
    PrimaryStream = MECHANICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 520000          # DMM: Copied from propulsion_ice_oil_bulk_carrier_panamax.inc
    OPEX = <520000 * 0.01>  # DMM: Copied from propulsion_ice_oil_bulk_carrier_panamax.inc
}

Converter "electrical_ice_diesel_bulk_carrier_panamax" {
    PowerCapacity=2.
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0.5, 0.5]    # DMM: Copied from electrical_ice_oil_bulk_carrier_panamax.inc
    PrimaryStream = ELECTRICAL
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 400000      # DMM: Copied from electrical_ice_oil_bulk_carrier_panamax.inc
    OPEX = 40000        # DMM: Copied from electrical_ice_oil_bulk_carrier_panamax.inc
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

Converter "heat_boiler_oil_diesel_bulk_carrier_panamax" {
    PowerCapacity=2.
    MainFuelTypes = [DIESEL]
    OutStream = [0, 0, 0, 1.0]
    PrimaryStream = HEAT
    
    CAPEX = 55e3
    OPEX = 0
}

