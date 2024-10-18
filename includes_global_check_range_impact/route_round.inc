Route "australia_USA_round" {
    RouteType=ROUND_TRIP
    
    Ports=[Port("Australia"), Port("Japan"), Port("United States"), Port("Japan")]
    Distances=[14000, 4700, 4700, 4000]
    #Distances=[1, 1, 1, 1]
    Speeds=[17, 17, 17, 17]
    #CapacityUtilization=[0.6, 0.6, 0.65, 0.6]
    PortDurations=[3, 4, 5, 4]     # DMM: Time spent in port (days)
}
