# Definition of the fuel oil vessel and its power system.
Vessel "fuel_oil_vessel" {
    PowerSystem = PowerSystem("fuel_oil_power_system")
    Route = Route("vessel_route")
    Finance = Finance("vessel_finance")
    NominalCapacity = 8000
    Tanks = [Tank("fuel_oil_tank_main")]
    PropulsionLoadLevel = 10
}

# Same as in tutorial 1.
PowerSystem "fuel_oil_power_system" {
    Propulsion = [Converter("fuel_oil_main_engine")]
    Electrical = [Converter("fuel_oil_auxiliary_engine")]
    Heat = [Converter("fuel_oil_boiler")]
    PropulsionInStreams = Array [
        [0, 1.0, 0, 0]
    ]
    ElectricalInStreams = Array [
        [0, 0, 1.0, 0]
    ]
    HeatInStreams = Array [
        [0, 0, 0, 0.5]
    ]
}

Converter "fuel_oil_main_engine" {
    PowerCapacity = 50
    MainFuelTypes = HFO
    OutStream = [0, 0.51, 0, 0.49]
}

Converter "fuel_oil_auxiliary_engine" {
    PowerCapacity = 10
    MainFuelTypes = HFO
    OutStream = [0, 0, 0.5, 0.5]
}
 
Converter "fuel_oil_boiler" {
    PowerCapacity = 5
    MainFuelTypes = HFO
    OutStream = [0, 0, 0, 1.0]
}
