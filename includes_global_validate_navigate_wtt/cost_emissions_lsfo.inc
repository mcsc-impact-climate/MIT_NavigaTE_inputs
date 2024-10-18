# Definition of cost and emissions of LSFO in baseline scenario.

Port "port" {
    
# Emissions
#set_bunker_WTT_overwrite("low_sulfur_fuel_oil", "carbon_dioxide", 0.56)     # From GREET 2022
#set_bunker_TTW_overwrite("low_sulfur_fuel_oil", "carbon_dioxide", 3.23)     # From GREET 2022

#set_bunker_price_overwrite("low_sulfur_fuel_oil", Forecast("crude_oil_cost_global_2022"))

}


