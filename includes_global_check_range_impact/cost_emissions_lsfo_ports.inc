# Definition of cost and emissions of LSFO in baseline scenario.

Port "Australia" {
    
# Emissions
set_bunker_WTT_overwrite("low_sulfur_fuel_oil", "carbon_dioxide", 0.56)     # From GREET 2022

set_bunker_price_overwrite("low_sulfur_fuel_oil", Forecast("crude_oil_cost_global_2022"))

}

Port "Japan" {
    
# Emissions
set_bunker_WTT_overwrite("low_sulfur_fuel_oil", "carbon_dioxide", 0.7)     # From GREET 2022

set_bunker_price_overwrite("low_sulfur_fuel_oil", Forecast("crude_oil_cost_global_2022"))

}

Port "United States" {
    
# Emissions
set_bunker_WTT_overwrite("low_sulfur_fuel_oil", "carbon_dioxide", 0.4)     # From GREET 2022

set_bunker_price_overwrite("low_sulfur_fuel_oil", Forecast("crude_oil_cost_global_2022"))

}


