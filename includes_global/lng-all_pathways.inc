# Definition of lng


Fuel "lng-fossil" {
}

Fuel "lng-*" {
    FuelType = METHANE
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 48.6    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.428     # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methane"))
}
