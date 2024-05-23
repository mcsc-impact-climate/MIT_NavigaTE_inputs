# Definition of cost and emissions for electrolytic hydrogen using the US grid

Port "port" {
    
# Emissions
set_bunker_WTT_overwrite("hydrogen_electro", "carbon_dioxide", 0.74)

set_bunker_price_overwrite("hydrogen_electro", 3700)

}


