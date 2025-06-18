# Details from https://www.dnv.com/news/imo-mepc-83-ghg-requirements-approved-taking-effect-from-2028/

Regulation "net_zero_regulation_imo_tier2" {
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
    RemedialCost = 380
    
    # SharedThreshold units: kg emissions per GJ fuel
    SharedThreshold = <93.3 * Forecast("regulation_reduction_tier2")>
    set_include_vessel("*", TRUE)
    
    }

Forecast "regulation_reduction_tier2" {
    Table [
        "10-01-2025"    1.0
        "01-01-2028"    0.96
        "01-01-2029"    0.94
        "01-01-2030"    0.92
        "01-01-2031"    0.876
        "01-01-2032"    0.832
        "01-01-2033"    0.788
        "01-01-2034"    0.744
        "01-01-2035"    0.7
        "01-01-2040"    0.35
        "01-01-2050"    0.0
    ]
    Interpolate = LINEAR
}


