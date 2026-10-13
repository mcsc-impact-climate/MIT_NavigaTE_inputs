# Definition of LEO biofuel


Fuel "bio_leo-bio" {
}

Fuel "bio_leo-*" {
    FuelType = HFO
    FuelPath = BLACK
    FuelOrigin = FOSSIL
    
    LowerHeatingValue = 21.5846927    # Units: MJ/kg. Source: SI to Li et al, 2022 (https://pubs.acs.org/doi/suppl/10.1021/acs.est.2c03960/suppl_file/es2c03960_si_001.pdf)
    MassDensity = 1.25     # Units: MJ/kg. Source: SI to Li et al, 2022 (https://pubs.acs.org/doi/suppl/10.1021/acs.est.2c03960/suppl_file/es2c03960_si_001.pdf)
    
    set_TTW("carbon_dioxide", 2.00)         # Units: ton CO2 / ton fuel. Source: SI to Masum et al, 2023 (https://pubs.acs.org/doi/suppl/10.1021/acs.est.3c00388/suppl_file/es3c00388_si_001.pdf)
    set_TTW("methane", 8.63e-5)         # Units: ton CH4 / ton fuel. Source: SI to Masum et al, 2023 (https://pubs.acs.org/doi/suppl/10.1021/acs.est.3c00388/suppl_file/es3c00388_si_001.pdf)
    set_TTW("nitrous_oxide", 2.16e-5)      # Units: ton N2O / ton fuel. Source: SI to Masum et al, 2023 (https://pubs.acs.org/doi/suppl/10.1021/acs.est.3c00388/suppl_file/es3c00388_si_001.pdf)
