Fuel "methanol" {
    FuelType = METHANOL
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    
    LowerHeatingValue = 20.1    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.794         # Units: kg/L. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methanol"))      # From default NavigaTE inputs
}
