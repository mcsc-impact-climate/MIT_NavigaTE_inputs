# Definition of cost and emissions foor ammonia

Port "port" {
    
# Emissions
set_bunker_WTT_overwrite("ammonia_electro", "carbon_dioxide", 3.75)
set_bunker_WTT_overwrite("ammonia_blue", "carbon_dioxide", 3.75)

set_bunker_price_overwrite("ammonia_electro", 1750)

}


