# Definition of cost and emissions for blue ammonia from the US using SMR+CCS

Fuel "methanol_blue" {
    FuelType = METHANOL
    FuelPath = BLUE
    FuelOrigin = FOSSIL_ABATED
    LowerHeatingValue = Variable("lower_heating_value_methanol")
    MassDensity = Variable("mass_density_methanol")
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methanol"))
}



