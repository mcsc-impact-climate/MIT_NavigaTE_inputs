# Definition of blue ammonia
Fuel "ammonia_blue" {
    FuelType = AMMONIA
    FuelPath = BLUE
    IsBunkerFuel = TRUE
    Process = Process("haber_bosch_ccs_integrated_blue")
    LowerHeatingValue = 18.8
    MassDensity = 0.623
}


Process "haber_bosch_ccs_integrated_blue" {
    Feedstocks=[Feedstock("natural_gas"), Feedstock("nitrogen")]
    Conversions=[0.64, 0.822]
}


Feedstock "natural_gas" {
}

Feedstock "nitrogen" {
}
