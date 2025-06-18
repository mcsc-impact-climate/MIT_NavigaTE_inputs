# Definition of hydrogen


Fuel "liquid_hydrogen-ATRCCS_H_grid_E" {
}
Fuel "liquid_hydrogen-ATRCCS_H_solar_E" {
}
Fuel "liquid_hydrogen-ATRCCS_H_wind_E" {
}
Fuel "liquid_hydrogen-BG_H_grid_E" {
}
Fuel "liquid_hydrogen-BG_H_solar_E" {
}
Fuel "liquid_hydrogen-BG_H_wind_E" {
}
Fuel "liquid_hydrogen-LTE_H_grid_E" {
}
Fuel "liquid_hydrogen-LTE_H_solar_E" {
}
Fuel "liquid_hydrogen-LTE_H_wind_E" {
}
Fuel "liquid_hydrogen-SMRCCS_H_grid_E" {
}
Fuel "liquid_hydrogen-SMRCCS_H_solar_E" {
}
Fuel "liquid_hydrogen-SMRCCS_H_wind_E" {
}
Fuel "liquid_hydrogen-SMR_H_grid_E" {
}
Fuel "liquid_hydrogen-SMR_H_solar_E" {
}
Fuel "liquid_hydrogen-SMR_H_wind_E" {
}

Fuel "liquid_hydrogen-*" {
    FuelType = HYDROGEN
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = 120.0   # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.0708  # Units: kg/L. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
}
