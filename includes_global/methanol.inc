Fuel "methanol" {
    FuelType = METHANOL
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    
    LowerHeatingValue = 19.9    # Units: MJ/kg. From https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.308(73).pdf
    MassDensity = 0.792         # Units: kg/L. From https://pubchem.ncbi.nlm.nih.gov/compound/Methanol#section=Density&fullscreen=true (at 20C)
    
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methanol"))
}
