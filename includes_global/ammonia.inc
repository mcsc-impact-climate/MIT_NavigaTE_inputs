# Definition of ammonia

Fuel "ammonia" {
    FuelType = AMMONIA
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = 18.8        # Units: MJ/kg. From GREET1_2023 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.6818            # Units: kg/L. From https://pubchem.ncbi.nlm.nih.gov/compound/Ammonia#section=Density&fullscreen=true (at boiling point of -33.35C)
    # set_TTW("nitrous_oxide", 0.00018)   # DMM: Normally defined elsewhere
}
