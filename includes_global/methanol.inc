Fuel "methanol" {
    FuelType = METHANOL
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    
    LowerHeatingValue = 20.1    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.794         # Units: kg/L. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", 1.392)        # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("nitrous_oxide", 7.799e-5)      # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("methane", 2.291e-5)            # Units: ton CO2 / ton fuel. From GREET 2024
}
