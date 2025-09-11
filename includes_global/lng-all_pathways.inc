# Definition of lng


Fuel "lng-ATRCCS_H_BEC_C_grid_E" {
}
Fuel "lng-ATRCCS_H_BEC_C_solar_E" {
}
Fuel "lng-ATRCCS_H_BEC_C_wind_E" {
}
Fuel "lng-ATRCCS_H_DAC_C_grid_E" {
}
Fuel "lng-ATRCCS_H_DAC_C_solar_E" {
}
Fuel "lng-ATRCCS_H_DAC_C_wind_E" {
}
Fuel "lng-BG_H_BEC_C_grid_E" {
}
Fuel "lng-BG_H_BEC_C_solar_E" {
}
Fuel "lng-BG_H_BEC_C_wind_E" {
}
Fuel "lng-BG_H_C_grid_E" {
}
Fuel "lng-BG_H_C_solar_E" {
}
Fuel "lng-BG_H_C_wind_E" {
}
Fuel "lng-BG_H_DAC_C_grid_E" {
}
Fuel "lng-BG_H_DAC_C_solar_E" {
}
Fuel "lng-BG_H_DAC_C_wind_E" {
}
Fuel "lng-LTE_H_BEC_C_grid_E" {
}
Fuel "lng-LTE_H_BEC_C_solar_E" {
}
Fuel "lng-LTE_H_BEC_C_wind_E" {
}
Fuel "lng-LTE_H_DAC_C_grid_E" {
}
Fuel "lng-LTE_H_DAC_C_solar_E" {
}
Fuel "lng-LTE_H_DAC_C_wind_E" {
}
Fuel "lng-SMRCCS_H_BEC_C_grid_E" {
}
Fuel "lng-SMRCCS_H_BEC_C_solar_E" {
}
Fuel "lng-SMRCCS_H_BEC_C_wind_E" {
}
Fuel "lng-SMRCCS_H_DAC_C_grid_E" {
}
Fuel "lng-SMRCCS_H_DAC_C_solar_E" {
}
Fuel "lng-SMRCCS_H_DAC_C_wind_E" {
}
Fuel "lng-SMR_H_BEC_C_grid_E" {
}
Fuel "lng-SMR_H_BEC_C_solar_E" {
}
Fuel "lng-SMR_H_BEC_C_wind_E" {
}
Fuel "lng-SMR_H_C_grid_E" {
}
Fuel "lng-SMR_H_C_solar_E" {
}
Fuel "lng-SMR_H_C_wind_E" {
}
Fuel "lng-SMR_H_DAC_C_grid_E" {
}
Fuel "lng-SMR_H_DAC_C_solar_E" {
}
Fuel "lng-SMR_H_DAC_C_wind_E" {
}
Fuel "lng-fossil" {
}

Fuel "lng-*" {
    FuelType = METHANE
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 48.6    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.428     # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", 2.6244)      # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("nitrous_oxide", 9.234e-5)      # Units: ton CO2 / ton fuel. From GREET 2024
    # Note: methane is accounted for in NavigaTE/defaults/user/Forecast/methane_slip_*.inc
}
