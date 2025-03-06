Vessel "bulk_carrier_handy_ice_liquid_hydrogen" {
    
    PropulsionLoadLevel = Surface("speed_draft_power_surface_bulk_carrier_handy")
    ElectricalLoadLevelAtSea=0.3
    ElectricalLoadLevelInPort=0.5
    HeatLoadLevelAtSea=0.25
    HeatLoadLevelInPort=0.49
    
    PowerSystem=PowerSystem("ice_hydrogen_bulk_carrier_handy")   # DMM: This will need to change for hydrogen
    Tanks = [Tank("main_liquid_hydrogen_bulk_carrier_handy"), Tank("pilot_oil_bulk_carrier_capesize")]
    
    Route=Route("globalized_bulk_carrier_handy")
    
    NominalCapacity = 10688.49467271139

    Lifetime=25
    Finance=Finance("vessel_finance")
    CAPEX = 23e6
    OPEX = 1.5e6
    
}

PowerSystem "ice_hydrogen_bulk_carrier_handy" {
    Propulsion = [Converter("propulsion_ice_hydrogen_bulk_carrier_handy")]
    Electrical = [Converter("electrical_ice_hydrogen_bulk_carrier_handy")]
    Heat = [Converter("heat_boiler_oil_hydrogen_bulk_carrier_handy")]
    
    PropulsionInStreams = Array [
        [0, 1.0, 0, 0]
    ]
    
    ElectricalInStreams = Array [
        [0, 0, 1.0, 0]
    ]
    
    HeatInStreams = Array [
        [0, 0, 0, 0.825]
    ]
    
    CAPEX = 2.92e6
    OPEX = 2.92e4
}

Converter "propulsion_ice_hydrogen_bulk_carrier_handy" {
    PowerCapacity=6.3
    MainFuelTypes = [HYDROGEN]
    PilotFuelTypes = [DIESEL, HFO]
    MinimumPilotFuel = 0.05
    OutStream = [0, Forecast("propulsion_ice_hydrogen_thermal_efficiency"), 0, <1 - Forecast("propulsion_ice_hydrogen_thermal_efficiency")>]
    PrimaryStream = MECHANICAL
    
    set_consumption_TTW(HYDROGEN, "nitrous_oxide", 0.)
    set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 700000
    OPEX = <700000 * 0.015>
}

Converter "electrical_ice_hydrogen_bulk_carrier_handy" {
    PowerCapacity=1.7
    MainFuelTypes = [HYDROGEN]
    PilotFuelTypes = [DIESEL, HFO]
    MinimumPilotFuel = 0.05
    OutStream = [0, 0, 0.48, 0.52]
    PrimaryStream = ELECTRICAL
    
    set_consumption_TTW(HYDROGEN, "nitrous_oxide", 0.)
    set_consumption_TTW(HFO, "black_carbon", Variable("converter_hfo_black_carbon"))
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
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

Converter "heat_boiler_oil_hydrogen_bulk_carrier_handy" {
    PowerCapacity=2.
    MainFuelTypes = [DIESEL, HFO]
    OutStream = [0, 0, 0, 1.0]
    PrimaryStream = HEAT
    
    CAPEX = 55e3
    OPEX = 0
}

