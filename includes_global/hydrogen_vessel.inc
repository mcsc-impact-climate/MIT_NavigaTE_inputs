# Definition of hydrogen vessel and its power system.
Vessel "hydrogen_vessel" {
    PowerSystem = PowerSystem("hydrogen_power_system")
    Route = Route("vessel_route")
    Finance = Finance("vessel_finance")
    NominalCapacity = 8000
    Tanks = [Tank("fuel_oil_tank_pilot"), Tank("hydrogen_tank_main")]
    PropulsionLoadLevel = 10
}


PowerSystem "hydrogen_power_system" {
    Propulsion = [Converter("hydrogen_main_engine")]
    Electrical = [Converter("hydrogen_auxiliary_engine")]
    Heat = [Converter("hydrogen_boiler")]
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

Converter "hydrogen_main_engine" {
    PowerCapacity = 50
    MainFuelTypes = HYDROGEN
    PilotFuelTypes = HFO
    MinimumPilotFuel = 0.05
    OutStream = [0, 0.51, 0, 0.49]
}


Converter "hydrogen_auxiliary_engine" {
    PowerCapacity = 10
    MainFuelTypes = HYDROGEN
    PilotFuelTypes = HFO
    MinimumPilotFuel = 0.05
    OutStream = [0, 0, 0.5, 0.5]
}

Converter "hydrogen_boiler" {
    PowerCapacity = 5
    MainFuelTypes = HFO
    OutStream = [0, 0, 0, 1.0]
}
