# Definition of cost and emissions for blue ammonia from the US using SMR+CCS

Fuel "methanol_electro" {
    FuelType = METHANOL
    FuelPath = GREEN
    FuelOrigin = SUSTAINABLE
    LowerHeatingValue = Variable("lower_heating_value_methanol")
    MassDensity = Variable("mass_density_methanol")
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methanol"))
}



