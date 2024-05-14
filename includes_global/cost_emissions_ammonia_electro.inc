# Definition of cost and emissions for ammonia

Port "port" {
    
# Emissions
set_bunker_WTT_overwrite("ammonia_electro", "carbon_dioxide", 3.75)
#set_bunker_WTT_overwrite("ammonia_blue", "carbon_dioxide", 0.669993768)
#set_bunker_WTT_overwrite("ammonia_grey", "carbon_dioxide", 2.180765544)

set_bunker_price_overwrite("ammonia_electro", 1750)
#set_bunker_price_overwrite("ammonia_blue", 610)
#set_bunker_price_overwrite("ammonia_grey", 490)

}


