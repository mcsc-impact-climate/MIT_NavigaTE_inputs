# Definition of ammonia

Fuel "ammonia" {
    FuelType = AMMONIA
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = 18.8        # Units: MJ/kg. From GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.6028            # Units: kg/L. From GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("nitrous_oxide", 5.64e-7)      # Units: ton CO2 / ton fuel. From GREET 2024
}
