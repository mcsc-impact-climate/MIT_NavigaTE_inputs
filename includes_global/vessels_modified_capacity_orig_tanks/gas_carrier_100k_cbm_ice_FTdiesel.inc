# DMM: Copied from gas_carrier_100k_cbm_ice_oil.inc
Vessel "gas_carrier_100k_cbm_ice_FTdiesel" {
    
    PropulsionLoadLevel = Curve("speed_power_curve_gas_carrier_100k_cbm")
    ElectricalLoadLevelAtSea=0.6
    ElectricalLoadLevelInPort=2.4
    HeatLoadLevelAtSea=0.6
    HeatLoadLevelInPort=0.6
    
    PowerSystem=PowerSystem("ice_diesel_gas_carrier_100k_cbm")
    Tanks = [Tank("main_FTdiesel_gas_carrier_100k_cbm")]
    
    Route=Route("globalized_gas_carrier_100k_cbm")
    
    NominalCapacity = 99726.16641107858

    Lifetime=25
    CostOfCapital = Variable("cost_of_capital_vessel")
    CAPEX = 129e6
    OPEX = 2.9e6
    
}

PowerSystem "ice_diesel_gas_carrier_100k_cbm" {
    Setup = INDIVIDUAL
    Propulsion = Converter("propulsion_ice_diesel_gas_carrier_100k_cbm")
    Electrical = Converter("electrical_ice_diesel_gas_carrier_100k_cbm")
    Heat = Converter("heat_boiler_diesel_gas_carrier_100k_cbm")

    CAPEX = 0.325e6     # DMM: Copied from ice_oil_gas_carrier_100k_cbm.inc
    OPEX = 0.325e4      # DMM: Copied from ice_oil_gas_carrier_100k_cbm.inc
}

Converter "propulsion_ice_diesel_gas_carrier_100k_cbm" {
    PowerCapacity=12.
    MinimumLoad = 0.3
    MainFuelTypes = [DIESEL]
    Efficiency = Forecast("propulsion_ice_diesel_thermal_efficiency")
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 520000              # DMM: Copied from propulsion_ice_oil_gas_carrier_100k_cbm.inc
    OPEX = <520000 * 0.01>      # DMM: Copied from propulsion_ice_oil_gas_carrier_100k_cbm.inc
}

Converter "electrical_ice_diesel_gas_carrier_100k_cbm" {
    PowerCapacity=3.
    MainFuelTypes = [DIESEL]
    Efficiency = 0.5        # DMM: Copied from electrical_ice_oil_gas_carrier_100k_cbm.inc
    
    set_consumption_TTW(DIESEL, "black_carbon", Variable("converter_diesel_black_carbon"))
    
    CAPEX = 400000      # DMM: Copied from electrical_ice_oil_gas_carrier_100k_cbm.inc
    OPEX = 40000        # DMM: Copied from electrical_ice_oil_gas_carrier_100k_cbm.inc
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
