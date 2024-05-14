# Definition of blue ammonia
Fuel "ammonia_blue" {
    FuelType = AMMONIA
    FuelPath = BLUE
    IsBunkerFuel = TRUE
    Process = Process("NH3_SMR_CCS_USA_GRID")
    LowerHeatingValue = 18.8
    MassDensity = 0.623
}


Process "NH3_SMR_CCS_USA_GRID" {
    Feedstocks=[Feedstock("ammonia_blue_ccs_usa_grid")]
    Conversions=[1]
}


Feedstock "ammonia_blue_ccs_usa_grid" {
}
