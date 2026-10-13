# Definition of bio-oil from catalytic fast pyrolysis with ZSM-5 catalyst
# All specs from GREET 2024 (https://greet.anl.gov/greet_1_series) unless otherwise indicated


Fuel "bio_cfp-bio" {
}

Fuel "bio_cfp-*" {
    FuelType = HFO
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 31.7374694    # Units: MJ/kg. Source: SI to Li et al, 2022 (https://pubs.acs.org/doi/suppl/10.1021/acs.est.2c03960/suppl_file/es2c03960_si_001.pdf)
    MassDensity = 0.97     # Units: MJ/kg. Source: SI to Li et al, 2022 (https://pubs.acs.org/doi/suppl/10.1021/acs.est.2c03960/suppl_file/es2c03960_si_001.pdf)
    
    set_TTW("carbon_dioxide", 2.79)         # Units: ton CO2 / ton fuel. From GREET 2024
    set_TTW("methane", 1.23e-4)         # Units: ton CH4 / ton fuel. From GREET 2024
    set_TTW("nitrous_oxide", 7.71e-5)      # Units: ton N2O / ton fuel. From GREET 2024
}
