Fuel "methanol_grey" {
    FuelType = METHANOL
    FuelPath = GREY
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = Variable("lower_heating_value_methanol")
    MassDensity = Variable("mass_density_methanol")
    
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methanol"))
}
