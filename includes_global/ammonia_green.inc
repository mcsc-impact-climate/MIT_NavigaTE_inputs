# Definition of green ammonia
Fuel "ammonia_green" {
    FuelType = AMMONIA
    FuelPath = GREEN
    IsBunkerFuel = TRUE
    Process = Process("ELECTROLYSIS_US_GRID")
    LowerHeatingValue = 18.8
    MassDensity = 0.623
}


Process "ELECTROLYSIS_US_GRID" {
    Feedstocks=[Feedstock("ammonia_green_us_grid")]
    Conversions=[1]
}

Feedstock "ammonia_green_us_grid" {
}
