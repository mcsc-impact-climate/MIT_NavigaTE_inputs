# Definition of cost and emissions of LSFO in baseline scenario.

Port "route_port" {

# Costs
set_feedstock_cost("crude_oil", 600)
set_feedstock_WTT("crude_oil", "carbon_dioxide", 0.3)
set_feedstock_WTT("crude_oil", "methane", 0.05)
    
# Emissions
set_process_WTT("fuel_oil_process", "carbon_dioxide", 0.68)

}
