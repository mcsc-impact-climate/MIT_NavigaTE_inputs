# Definition of low_sulfur_fuel_oil

Fuel "lng" {
    FuelType = HFO
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 48.6    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.428     # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", 2.75)      # From default NavigaTE inputs
    set_TTW("nitrous_oxide", 0.00011)      # From default NavigaTE inputs
}
