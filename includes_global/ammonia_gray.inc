# Definition of grey ammonia
Fuel "ammonia_grey" {
    FuelType = AMMONIA
    FuelPath = GREY
    IsBunkerFuel = TRUE
    Process = Process("SMR")
    LowerHeatingValue = 18.8    #GJ/ton
    MassDensity = 0.623
}


Process "SMR" {
    Feedstocks=[Feedstock("ammonia_grey_usa_grid")]
    Conversions=[1]
}


Feedstock "ammonia_grey_usa_grid" {
}
