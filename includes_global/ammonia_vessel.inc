# Definition of ammonia vessel and its power system.
Vessel "ammonia_vessel" {
    PowerSystem = PowerSystem("ammonia_power_system")
    Route = Route("vessel_route")
    Finance = Finance("vessel_finance")
    NominalCapacity = 8000
    Tanks = [Tank("fuel_oil_tank_pilot"), Tank("ammonia_tank_main")]
    PropulsionLoadLevel = 10
}


PowerSystem "ammonia_power_system" {
    Propulsion = [Converter("ammonia_main_engine")]
    Electrical = [Converter("ammonia_auxiliary_engine")]
    Heat = [Converter("ammonia_boiler")]
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

Converter "ammonia_main_engine" {
    PowerCapacity = 50
    MainFuelTypes = AMMONIA
    PilotFuelTypes = HFO
    MinimumPilotFuel = 0.05
    OutStream = [0, 0.51, 0, 0.49]
}


Converter "ammonia_auxiliary_engine" {
    PowerCapacity = 10
    MainFuelTypes = AMMONIA
    PilotFuelTypes = HFO
    MinimumPilotFuel = 0.05
    OutStream = [0, 0, 0.5, 0.5]
}

Converter "ammonia_boiler" {
    PowerCapacity = 5
    MainFuelTypes = HFO
    OutStream = [0, 0, 0, 1.0]
}
