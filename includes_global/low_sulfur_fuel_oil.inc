# Definition of low_sulfur_fuel_oil

Fuel "low_sulfur_fuel_oil" {
    FuelType = HFO
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    LowerHeatingValue = 40.2    # Units: MJ/kg. Source: https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.308(73).pdf
    MassDensity = 0.991     # Units: MJ/kg. Source: https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.308(73).pdf
    set_TTW("carbon_dioxide", 3.114)
    set_TTW("methane", 0.00005)
    set_TTW("nitrous_oxide", 0.00018)
}
