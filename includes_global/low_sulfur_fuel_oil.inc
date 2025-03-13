# Definition of low_sulfur_fuel_oil

Fuel "low_sulfur_fuel_oil" {
    FuelType = HFO
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 39.5    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.991     # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", 3.114)      # From default NavigaTE inputs
    set_TTW("methane", 0.00005)      # From default NavigaTE inputs
    set_TTW("nitrous_oxide", 0.00018)      # From default NavigaTE inputs
}
