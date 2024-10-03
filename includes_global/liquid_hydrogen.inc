# Definition of hydrogen

Fuel "liquid_hydrogen" {
    FuelType = HYDROGEN
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = Variable("lower_heating_value_hydrogen")
    MassDensity = Variable("mass_density_methanol")  # kg/L 
    # set_TTW("nitrous_oxide", 0.00018)   # DMM: Normally defined elsewhere
}
