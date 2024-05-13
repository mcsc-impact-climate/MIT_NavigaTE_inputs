# Definition of costs and emissions of ammonia

Port "route_port" {

# Costs
set_feedstock_cost("natural_gas", 250)
set_process_OPEX("haber_bosch_ccs_integrated_blue", 100)

# Emissions
set_feedstock_WTT("natural_gas", "carbon_dioxide", 0.11)
set_feedstock_WTT("natural_gas", "methane", 0.0073)
set_process_WTT("haber_bosch_ccs_integrated_blue", "carbon_dioxide", 0.180)

}
