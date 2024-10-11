# Definition of hydrogen

Fuel "hydrogen" {
    FuelType = HYDROGEN
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = Variable("lower_heating_value_hydrogen")
    MassDensity = 0.083
    # set_TTW("nitrous_oxide", 0.00018)   # DMM: Normally defined elsewhere
}
