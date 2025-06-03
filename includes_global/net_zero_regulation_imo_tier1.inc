# Details from https://www.dnv.com/news/imo-mepc-83-ghg-requirements-approved-taking-effect-from-2028/

Regulation "net_zero_regulation_imo_tier1" {
    Measure = INTENSITY
    Scheme = FLEXIBLE
    
    Scope = WTW
    Unit = GWP
    IncludeSlip = TRUE
    
    EmissionTarget = ALL
    FuelTarget = ALL
    
    Jurisdiction = [Port("port_singapore"), Port("port_rotterdam")]
    IntraFraction = 1.
    InterFraction = 1.
    ExtraFraction = 0.
    
    # Penalty to be paid for breaching emissions threshold (USD / tonne CO2e)
    RemedialCost = 100
    
    # SharedThreshold units: kg emissions per GJ fuel
    SharedThreshold = <93.3 * Forecast("regulation_reduction_tier1")>
    set_include_vessel("*", TRUE)
    
    }

Forecast "regulation_reduction_tier1" {
    Table [
        "10-01-2025"    1.0
        "01-01-2028"    0.83
        "01-01-2029"    0.81
        "01-01-2030"    0.79
        "01-01-2031"    0.746
        "01-01-2032"    0.702
        "01-01-2033"    0.658
        "01-01-2034"    0.614
        "01-01-2035"    0.57
        "01-01-2040"    0.3
        "01-01-2050"    0.0
    ]
    Interpolate = LINEAR
}


