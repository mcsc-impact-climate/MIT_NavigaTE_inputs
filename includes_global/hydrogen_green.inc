# Definition of blue hydrogen
Fuel "hydrogen_blue" {
    FuelType = HYDROGEN
    FuelPath = GREEN
    IsBunkerFuel = TRUE
    Process = Process("H2_SMR_USA_GRID")
    LowerHeatingValue = 120.1
    MassDensity = 70.85     # From https://en.wikipedia.org/wiki/Liquid_hydrogen
}


Process "H2_SMR_USA_GRID" {
    Feedstocks=[Feedstock("hydrogen_green_usa_grid")]
    Conversions=[1]
}


Feedstock "hydrogen_green_usa_grid" {
}
