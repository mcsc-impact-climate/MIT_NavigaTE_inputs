# Definition of cost and emissions for ammonia (US wind)

Port "port" {
    
# Emissions
set_bunker_WTT_overwrite("ammonia_electro", "carbon_dioxide", 0.13)

# Price
set_bunker_price_overwrite("ammonia_electro", 610)
}


