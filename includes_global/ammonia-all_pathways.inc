# Definition of ammonia


Fuel "ammonia-ATRCCS_H_grid_E" {
}
Fuel "ammonia-ATRCCS_H_solar_E" {
}
Fuel "ammonia-ATRCCS_H_wind_E" {
}
Fuel "ammonia-BG_H_grid_E" {
}
Fuel "ammonia-BG_H_solar_E" {
}
Fuel "ammonia-BG_H_wind_E" {
}
Fuel "ammonia-LTE_H_grid_E" {
}
Fuel "ammonia-LTE_H_solar_E" {
}
Fuel "ammonia-LTE_H_wind_E" {
}
Fuel "ammonia-SMRCCS_H_grid_E" {
}
Fuel "ammonia-SMRCCS_H_solar_E" {
}
Fuel "ammonia-SMRCCS_H_wind_E" {
}
Fuel "ammonia-SMR_H_grid_E" {
}
Fuel "ammonia-SMR_H_solar_E" {
}
Fuel "ammonia-SMR_H_wind_E" {
}

Fuel "ammonia-*" {
    FuelType = AMMONIA
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = 18.8        # Units: MJ/kg. From GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.6028            # Units: kg/L. From GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("nitrous_oxide", 5.64e-7)      # Units: ton CO2 / ton fuel. From GREET 2024
}
