# Definition of Fischer Tropsch diesel

Fuel "FTdiesel" {
    FuelType = DIESEL
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = 41.2         # Units: MJ/kg. Source: https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.308(73).pdf
    MassDensity = 0.900       # Units: kg / L. Source: https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.308(73).pdf
    
    set_TTW("carbon_dioxide", 3.11)      # Units: ton CO2 / ton fuel, obtained from GREET 2022
    set_TTW("methane", 4.92e-5)         # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("nitrous_oxide", 0.000167)      # Units: ton CO2 / ton fuel. From GREET 2024
}
