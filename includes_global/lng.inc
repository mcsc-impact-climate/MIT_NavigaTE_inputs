# Definition of lng

Fuel "lng" {
    FuelType = METHANE
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 48.6    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.428     # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", 2.6244)      # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("nitrous_oxide", 9.234e-5)      # Units: ton CO2 / ton fuel. From GREET 2024
    # Note: methane is accounted for in NavigaTE/defaults/user/Forecast/methane_slip_*.inc
}
