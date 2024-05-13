# Definition of low_sulfur_fuel_oil

Fuel "low_sulfur_fuel_oil" {
    FuelType = HFO
    FuelPath = BLACK
    IsBunkerFuel = TRUE
    Process = Process("fuel_oil_process")
    LowerHeatingValue = 41.2
    MassDensity = 0.9
    set_TTW("carbon_dioxide", 3.114)
    set_TTW("methane", 0.00005)
    set_TTW("nitrous_oxide", 0.00018)
}

# Definition of "fuel_oil_process" (same as Tutorial 1).
Process "fuel_oil_process" {
    Feedstocks = [Feedstock("crude_oil")]
    Conversions = [1.0]
}

# Definition of "crude_oil" (same as Tutorial 1).
Feedstock "crude_oil" {
}
