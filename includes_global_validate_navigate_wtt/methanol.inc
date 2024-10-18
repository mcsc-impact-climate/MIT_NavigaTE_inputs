Fuel "methanol" {
    FuelType = METHANOL
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    
    LowerHeatingValue = Variable("lower_heating_value_methanol")
    MassDensity = Variable("mass_density_methanol")
    
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methanol"))
}
