# Definition of Fischer Tropsch diesel


Fuel "FTdiesel-ATRCCS_H_BEC_C_grid_E" {
}
Fuel "FTdiesel-ATRCCS_H_BEC_C_solar_E" {
}
Fuel "FTdiesel-ATRCCS_H_BEC_C_wind_E" {
}
Fuel "FTdiesel-ATRCCS_H_DAC_C_grid_E" {
}
Fuel "FTdiesel-ATRCCS_H_DAC_C_solar_E" {
}
Fuel "FTdiesel-ATRCCS_H_DAC_C_wind_E" {
}
Fuel "FTdiesel-BG_H_BEC_C_grid_E" {
}
Fuel "FTdiesel-BG_H_BEC_C_solar_E" {
}
Fuel "FTdiesel-BG_H_BEC_C_wind_E" {
}
Fuel "FTdiesel-BG_H_C_grid_E" {
}
Fuel "FTdiesel-BG_H_C_solar_E" {
}
Fuel "FTdiesel-BG_H_C_wind_E" {
}
Fuel "FTdiesel-BG_H_DAC_C_grid_E" {
}
Fuel "FTdiesel-BG_H_DAC_C_solar_E" {
}
Fuel "FTdiesel-BG_H_DAC_C_wind_E" {
}
Fuel "FTdiesel-LTE_H_BEC_C_grid_E" {
}
Fuel "FTdiesel-LTE_H_BEC_C_solar_E" {
}
Fuel "FTdiesel-LTE_H_BEC_C_wind_E" {
}
Fuel "FTdiesel-LTE_H_DAC_C_grid_E" {
}
Fuel "FTdiesel-LTE_H_DAC_C_solar_E" {
}
Fuel "FTdiesel-LTE_H_DAC_C_wind_E" {
}
Fuel "FTdiesel-SMRCCS_H_BEC_C_grid_E" {
}
Fuel "FTdiesel-SMRCCS_H_BEC_C_solar_E" {
}
Fuel "FTdiesel-SMRCCS_H_BEC_C_wind_E" {
}
Fuel "FTdiesel-SMRCCS_H_DAC_C_grid_E" {
}
Fuel "FTdiesel-SMRCCS_H_DAC_C_solar_E" {
}
Fuel "FTdiesel-SMRCCS_H_DAC_C_wind_E" {
}
Fuel "FTdiesel-SMR_H_BEC_C_grid_E" {
}
Fuel "FTdiesel-SMR_H_BEC_C_solar_E" {
}
Fuel "FTdiesel-SMR_H_BEC_C_wind_E" {
}
Fuel "FTdiesel-SMR_H_C_grid_E" {
}
Fuel "FTdiesel-SMR_H_C_solar_E" {
}
Fuel "FTdiesel-SMR_H_C_wind_E" {
}
Fuel "FTdiesel-SMR_H_DAC_C_grid_E" {
}
Fuel "FTdiesel-SMR_H_DAC_C_solar_E" {
}
Fuel "FTdiesel-SMR_H_DAC_C_wind_E" {
}

Fuel "FTdiesel-*" {
    FuelType = DIESEL
    FuelPath = USER_DEFINED
    FuelOrigin = USER_DEFINED
    LowerHeatingValue = 41.2         # Units: MJ/kg. Source: https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.308(73).pdf
    MassDensity = 0.900       # Units: kg / L. Source: https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MEPCDocuments/MEPC.308(73).pdf
    set_TTW("carbon_dioxide", 3.11)      # Units: ton CO2 / ton fuel, obtained from GREET 2022
}
