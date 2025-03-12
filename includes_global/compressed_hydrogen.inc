# Definition of hydrogen

Fuel "compressed_hydrogen" {
    FuelType = HYDROGEN
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = 120.0   # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.03873  # Units: kg / L for H2 at 700 bar and 20 C. Source: https://cmb.tech/hydrogen-tools
}
