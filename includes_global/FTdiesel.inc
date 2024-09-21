# Definition of Fischer Tropsch diesel

Fuel "FTdiesel" {
    FuelType = DIESEL
    FuelPath = GREEN
    FuelOrigin = SUSTAINABLE
    LowerHeatingValue = 43.2         # Units: GJ / tonne
    MassDensity = Variable("mass_density_diesel")       # Units: kg / L
    set_TTW("carbon_dioxide", 2.376)      # Units: ton CO2 / ton fuel, obtained from GREET 2022
}
