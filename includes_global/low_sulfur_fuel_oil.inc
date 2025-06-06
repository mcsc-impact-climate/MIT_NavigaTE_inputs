# Definition of low_sulfur_fuel_oil

Fuel "low_sulfur_fuel_oil" {
    FuelType = HFO
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 39.5    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.991     # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", 3.16)         # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("methane", 4.661e-5)         # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("nitrous_oxide", 0.000158)      # Units: ton CO2 / ton fuel. From GREET 2024
}
