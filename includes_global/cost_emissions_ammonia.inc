# Definition of costs and emissions of grey ammonia with the US grid

Port "route_port" {

# Costs
set_feedstock_cost("ammonia_grey_usa_grid", 490)  # USD/ton
set_feedstock_cost("ammonia_blue_ccs_usa_grid", 610)  # USD/ton
set_feedstock_cost("ammonia_green_us_grid", 1750)  # USD/ton

# Emissions
set_feedstock_WTT("ammonia_grey_usa_grid", "carbon_dioxide", 2.18)    # Tons of emissions per ton of fuel
set_feedstock_WTT("ammonia_blue_ccs_usa_grid", "carbon_dioxide", 0.67)    # Tons of emissions per ton of fuel
set_feedstock_WTT("ammonia_green_us_grid", "carbon_dioxide", 3.75)    # Tons of emissions per ton of fuel
}
