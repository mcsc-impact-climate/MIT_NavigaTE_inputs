# Definition of cost and emissions for ammonia (US grid)

Port "port" {
    
# Emissions
set_bunker_WTT_overwrite("ammonia_electro", "carbon_dioxide", 3.75)

# Price
set_bunker_price_overwrite("ammonia_electro", 1750)
}


