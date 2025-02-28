Regulation "net_zero_regulation" {
    Measure = INTENSITY
    Scheme = FLEXIBLE
    
    Scope = WTW
    Unit = GWP
    IncludeSlip = TRUE
    
    EmissionTarget = ALL
    FuelTarget = ALL
    
    Jurisdiction = [Port("port")]
    IntraFraction = 1.
    InterFraction = 1.
    ExtraFraction = 0.
    
    # Penalty to be paid for breaching emissions threshold (USD / tonne CO2e)
    RemedialCost = 1200
    
    # SharedThreshold units: kg emissions per GJ fuel
    SharedThreshold = <100 * Forecast("regulation_reduction")>
    set_include_vessel("*", TRUE)
    
    }

Forecast "regulation_reduction" {
    Table [
        "01-01-2024"    1.
        "01-01-2030"    0.9
        "01-01-2050"    0.1
    ]
    Interpolate = LINEAR
}


