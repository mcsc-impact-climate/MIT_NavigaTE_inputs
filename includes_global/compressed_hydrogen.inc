# Definition of hydrogen

Fuel "compressed_hydrogen" {
    FuelType = HYDROGEN
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = Variable("lower_heating_value_hydrogen")
    MassDensity = 0.03873  # kg / L for H2 at 700 bar and 20 C, from https://cmb.tech/hydrogen-tools
    # set_TTW("nitrous_oxide", 0.00018)   # DMM: Normally defined elsewhere
}
