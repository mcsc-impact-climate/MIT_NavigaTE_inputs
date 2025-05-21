
Fuel "methanol-ATRCCS_H_BEC_C_grid_E" {
}
Fuel "methanol-ATRCCS_H_BEC_C_solar_E" {
}
Fuel "methanol-ATRCCS_H_BEC_C_wind_E" {
}
Fuel "methanol-ATRCCS_H_C_grid_E" {
}
Fuel "methanol-ATRCCS_H_C_solar_E" {
}
Fuel "methanol-ATRCCS_H_C_wind_E" {
}
Fuel "methanol-ATRCCS_H_DAC_C_grid_E" {
}
Fuel "methanol-ATRCCS_H_DAC_C_solar_E" {
}
Fuel "methanol-ATRCCS_H_DAC_C_wind_E" {
}
Fuel "methanol-ATRCCS_H_SMRCCS_C_grid_E" {
}
Fuel "methanol-ATRCCS_H_SMRCCS_C_solar_E" {
}
Fuel "methanol-ATRCCS_H_SMRCCS_C_wind_E" {
}
Fuel "methanol-BG_H_ATRCCS_C_grid_E" {
}
Fuel "methanol-BG_H_ATRCCS_C_solar_E" {
}
Fuel "methanol-BG_H_ATRCCS_C_wind_E" {
}
Fuel "methanol-BG_H_BEC_C_grid_E" {
}
Fuel "methanol-BG_H_BEC_C_solar_E" {
}
Fuel "methanol-BG_H_BEC_C_wind_E" {
}
Fuel "methanol-BG_H_C_grid_E" {
}
Fuel "methanol-BG_H_C_solar_E" {
}
Fuel "methanol-BG_H_C_wind_E" {
}
Fuel "methanol-BG_H_DAC_C_grid_E" {
}
Fuel "methanol-BG_H_DAC_C_solar_E" {
}
Fuel "methanol-BG_H_DAC_C_wind_E" {
}
Fuel "methanol-BG_H_SMRCCS_C_grid_E" {
}
Fuel "methanol-BG_H_SMRCCS_C_solar_E" {
}
Fuel "methanol-BG_H_SMRCCS_C_wind_E" {
}
Fuel "methanol-LTE_H_ATRCCS_C_grid_E" {
}
Fuel "methanol-LTE_H_ATRCCS_C_solar_E" {
}
Fuel "methanol-LTE_H_ATRCCS_C_wind_E" {
}
Fuel "methanol-LTE_H_BEC_C_grid_E" {
}
Fuel "methanol-LTE_H_BEC_C_solar_E" {
}
Fuel "methanol-LTE_H_BEC_C_wind_E" {
}
Fuel "methanol-LTE_H_DAC_C_grid_E" {
}
Fuel "methanol-LTE_H_DAC_C_solar_E" {
}
Fuel "methanol-LTE_H_DAC_C_wind_E" {
}
Fuel "methanol-LTE_H_SMRCCS_C_grid_E" {
}
Fuel "methanol-LTE_H_SMRCCS_C_solar_E" {
}
Fuel "methanol-LTE_H_SMRCCS_C_wind_E" {
}
Fuel "methanol-SMRCCS_H_ATRCCS_C_grid_E" {
}
Fuel "methanol-SMRCCS_H_ATRCCS_C_solar_E" {
}
Fuel "methanol-SMRCCS_H_ATRCCS_C_wind_E" {
}
Fuel "methanol-SMRCCS_H_BEC_C_grid_E" {
}
Fuel "methanol-SMRCCS_H_BEC_C_solar_E" {
}
Fuel "methanol-SMRCCS_H_BEC_C_wind_E" {
}
Fuel "methanol-SMRCCS_H_C_grid_E" {
}
Fuel "methanol-SMRCCS_H_C_solar_E" {
}
Fuel "methanol-SMRCCS_H_C_wind_E" {
}
Fuel "methanol-SMRCCS_H_DAC_C_grid_E" {
}
Fuel "methanol-SMRCCS_H_DAC_C_solar_E" {
}
Fuel "methanol-SMRCCS_H_DAC_C_wind_E" {
}
Fuel "methanol-SMR_H_BEC_C_grid_E" {
}
Fuel "methanol-SMR_H_BEC_C_solar_E" {
}
Fuel "methanol-SMR_H_BEC_C_wind_E" {
}
Fuel "methanol-SMR_H_C_grid_E" {
}
Fuel "methanol-SMR_H_C_solar_E" {
}
Fuel "methanol-SMR_H_C_wind_E" {
}
Fuel "methanol-SMR_H_DAC_C_grid_E" {
}
Fuel "methanol-SMR_H_DAC_C_solar_E" {
}
Fuel "methanol-SMR_H_DAC_C_wind_E" {
}

Fuel "methanol-*" {
    FuelType = METHANOL
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    
    LowerHeatingValue = 20.1    # Units: MJ/kg. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    MassDensity = 0.794         # Units: kg/L. Source: GREET1_2024 (https://greet.anl.gov/greet_1_series)
    
    set_TTW("carbon_dioxide", Variable("carbon_dioxide_per_methanol"))      # From default NavigaTE inputs
}
