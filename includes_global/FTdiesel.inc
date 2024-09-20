# Definition of Fischer Tropsch diesel

Fuel "FTdiesel" {
    FuelType = DIESEL
    FuelPath = GREEN
    FuelOrigin = SUSTAINABLE
    LowerHeatingValue = Variable("lower_heating_value_diesel")       # Units: GJ / ton at 20 degrees C and 1 atm
    MassDensity = Variable("mass_density_diesel")       # Units: kg / L
    # set_TTW("carbon_dioxide", 3.114)  # DMM: Normally defined elsewhere
    # set_TTW("methane", 0.00005)
    # set_TTW("nitrous_oxide", 0.00018)
}
