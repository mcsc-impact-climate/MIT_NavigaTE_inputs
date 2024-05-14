# Definition of costs and emissions of blue hydrogen with the US grid

Port "route_port" {

# Costs
set_feedstock_cost("hydrogen_blue_usa_grid", 3740)  # USD/ton
set_feedstock_cost("hydrogen_grey_usa_grid", 3100)  # USD/ton
set_feedstock_cost("hydrogen_green_usa_grid", 9960)  # USD/ton

# Emissions
set_feedstock_WTT("hydrogen_blue_usa_grid", "carbon_dioxide", 6.05)    # Tons of emissions per ton of fuel
set_feedstock_WTT("hydrogen_grey_usa_grid", "carbon_dioxide", 12.36)    # Tons of emissions per ton of fuel
set_feedstock_WTT("hydrogen_green_usa_grid", "carbon_dioxide", 20.91)    # Tons of emissions per ton of fuel

}
