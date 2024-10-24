"""
Date: July 1, 2024
Purpose: Model validations to compare NavigaTE fleet model inputs and outputs with other sources
"""
# get_top_dir used for file organization
# pandas as pd used to organize data in table format
from common_tools import get_top_dir
import pandas as pd

# twenty-foot equivalent unit (cargo capacity)
TONNES_PER_TEU = 14  # Average tons of cargo in one TEU. Obtained from https://www.mpc-container.com/about-us/industry-terms/

#conversion units
MJ_PER_GJ = 1000
KG_PER_TONNE = 1000
G_PER_KG = 1000
TONNES_PER_KT = 1000

# Collect the lower heating value of HFO (in MJ/kg) from the relevant info file
top_dir = get_top_dir()
fuel_info_df = pd.read_csv(f"{top_dir}/info_files/fuel_info.csv").set_index("Fuel")
HFO_LHV = fuel_info_df.loc["lsfo", "Lower Heating Value (MJ / kg)"]

#vessels dictionary holds different container types:
vessels = {
    "bulk": [
        "bulk_carrier_capesize_ice",
        "bulk_carrier_handy_ice",
        "bulk_carrier_panamax_ice",
    ],
    "container": [
        "container_15000_teu_ice",
        "container_8000_teu_ice",
        "container_3500_teu_ice",
    ],
    "tanker": ["tanker_100k_dwt_ice", "tanker_300k_dwt_ice", "tanker_35k_dwt_ice"],
}

# maps the names for the vessel models to specific strings to be understandable to readers
#capesize: largest dry cargo ships
vessel_size_title = {
    "bulk_carrier_capesize_ice": "Capesize",
    "bulk_carrier_handy_ice": "Handy",
    "bulk_carrier_panamax_ice": "Panamax",
    "container_15000_teu_ice": "15,000 TEU",
    "container_8000_teu_ice": "8,000 TEU",
    "container_3500_teu_ice": "3,500 TEU",
    "tanker_100k_dwt_ice": "100k DWT",
    "tanker_300k_dwt_ice": "300k DWT",
    "tanker_35k_dwt_ice": "35k DWT",
}

vessel_size_number = {
    "bulk_carrier_capesize_ice": 2013,
    "bulk_carrier_handy_ice": 4186,
    "bulk_carrier_panamax_ice": 6384,
    "container_15000_teu_ice": 464,
    "container_8000_teu_ice": 1205,
    "container_3500_teu_ice": 3896,
    "tanker_100k_dwt_ice": 3673,
    "tanker_300k_dwt_ice": 866,
    "tanker_35k_dwt_ice": 8464,
}

# different types of fuel in categories
fuel_components = {
    "hydrogen": ["hydrogen (main)", "lsfo (pilot)"],
    "ammonia": ["ammonia (main)", "lsfo (pilot)"],
    "lsfo": ["lsfo (main)"],
}

# Number of vessels of each type (from IMO 4th GHG study, Table 7)
vessel_numbers_imo = {"bulk": 11672, "container": 5182, "tanker": 13003}

#parameters: file name and lsfo fuel (Low Sulphur Fuel Oil heavy fuel oils)
def read_results(filename, fuel="lsfo"):
    columns = [
        "Vessel",
        "Fuel",
        "WTT Emissions (tonnes CO2 / year)",
        "TTW Emissions (tonnes CO2 / year)",
        "WTW Emissions (tonnes CO2 / year)",
        "CAPEX (USD / year)",
        "Fuel Cost (USD / year)",
        "Other OPEX (USD / year)",
        "Total Cost (USD / year)",
        "Energy Consumed (GJ / year) [main]",
        "Energy Consumed (GJ / year) [pilot]",
        "Energy Spend (GJ / year)",
        "Miles / year",
        "Cargo tonne-miles / year",
    ]
# sets up predetermined columns
    all_results_df = pd.DataFrame(columns=columns)
# use different sets of columns for the different fuels
    if fuel == "lsfo":
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles",
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            "ConsumedEnergy_lsfo",
        ]
    else:
        results_df_columns = [
            "Date",
            "Time (days)",
            "TotalEquivalentWTT",
            "TotalEquivalentTTW",
            "TotalEquivalentWTW",
            "Miles",
            "CargoMiles",
            "SpendEnergy",
            "TotalCAPEX",
            "TotalExcludingFuelOPEX",
            "TotalFuelOPEX",
            f"ConsumedEnergy_{fuel}",
            "ConsumedEnergy_lsfo",
        ]
# reads excel that's named "Vessels"
    results = pd.ExcelFile(filename)
    results_df = pd.read_excel(results, "Vessels")
# iterates over each vessel type
    for vessel_type in vessels:
        # iterates over each vessel in the vessel type
        for vessel in vessels[vessel_type]:
            results_dict = {}
            results_df_vessel = results_df.filter(regex=f"Date|Time|{vessel}").drop(
                [0, 1, 2]
            )
            results_df_vessel.columns = results_df_columns
            results_df_vessel = results_df_vessel.set_index("Date")
            #extracts info from the excel sheet
            results_dict["Vessel"] = f"{vessel}_{fuel}"
            results_dict["Fuel"] = fuel
            results_dict["WTT Emissions (tonnes CO2 / year)"] = results_df_vessel[
                "TotalEquivalentWTT"
            ].loc["2024-01-01"]
            results_dict["TTW Emissions (tonnes CO2 / year)"] = results_df_vessel[
                "TotalEquivalentTTW"
            ].loc["2024-01-01"]
            results_dict["WTW Emissions (tonnes CO2 / year)"] = results_df_vessel[
                "TotalEquivalentWTW"
            ].loc["2024-01-01"]
            results_dict["CAPEX (USD / year)"] = results_df_vessel["TotalCAPEX"].loc[
                "2024-01-01"
            ]
            results_dict["Fuel Cost (USD / year)"] = results_df_vessel[
                "TotalFuelOPEX"
            ].loc["2024-01-01"]
            results_dict["Other OPEX (USD / year)"] = results_df_vessel[
                "TotalExcludingFuelOPEX"
            ].loc["2024-01-01"]
            # calculates total cost per year (capital expenditures + operating expenditures of fuel + OPEX non fuel) 
            results_dict["Total Cost (USD / year)"] = (
                results_df_vessel["TotalCAPEX"].loc["2024-01-01"]
                + results_df_vessel["TotalFuelOPEX"].loc["2024-01-01"]
                + results_df_vessel["TotalExcludingFuelOPEX"].loc["2024-01-01"]
            )
            results_dict["Energy Spend (GJ / year)"] = results_df_vessel[
                "SpendEnergy"
            ].loc["2024-01-01"]
            results_dict["Miles / year"] = results_df_vessel["Miles"].loc["2024-01-01"]
            # difference of cargo tonne-mile/year depending on vessel type
            if vessel_type == "container":
                results_dict["Cargo tonne-miles / year"] = (
                    results_df_vessel["CargoMiles"].loc["2024-01-01"] * TONNES_PER_TEU
                )
            else:
                results_dict["Cargo tonne-miles / year"] = results_df_vessel[
                    "CargoMiles"
                ].loc["2024-01-01"]

            if fuel == "lsfo":
                results_dict["Energy Consumed (GJ / year) [main]"] = results_df_vessel[
                    "ConsumedEnergy_lsfo"
                ].loc["2024-01-01"]
                results_dict["Energy Consumed (GJ / year) [pilot]"] = (
                    results_df_vessel["ConsumedEnergy_lsfo"].loc["2024-01-01"] * 0
                )
            else:
                results_dict["Energy Consumed (GJ / year) [main]"] = results_df_vessel[
                    f"ConsumedEnergy_{fuel}"
                ].loc["2024-01-01"]
                results_dict["Energy Consumed (GJ / year) [pilot]"] = results_df_vessel[
                    "ConsumedEnergy_lsfo"
                ].loc["2024-01-01"]

            results_row_df = pd.DataFrame([results_dict])

            all_results_df = pd.concat(
                [all_results_df, results_row_df], ignore_index=True
            )
    # return the DataFrame containing all the vessels info
    return all_results_df


# Convert energy consumption to fuel consumption for LSFO
# Assume energy consumption is in GJ
# Fuel consumption is in 1000's of tonnes
# (unit conversions)
def energy_to_fuel_consumption_lsfo(energy_consumption_GJ):
    energy_consumption_MJ = energy_consumption_GJ * MJ_PER_GJ
    fuel_consumption_kg = energy_consumption_MJ / HFO_LHV
    fuel_consumption_thou_tonnes = fuel_consumption_kg / (KG_PER_TONNE * 1000)
    return fuel_consumption_thou_tonnes

# Compare annual fuel consumption evaluated for each vessel type between NavigaTE and IMO
def compare_fuel_consumption(all_results_df, fuel="lsfo"):
    # Total HFO-equivalent fuel consumption for each vessel type in thousand tonnes (from IMO 4th GHG study, Figure 5)
    annual_fuel_consumption_imo = {"bulk": 54359, "container": 63906, "tanker": 74674}

    columns = [
        "Vessel",
        "Global annual fuel cons (NavigaTE)",
        "Global annual fuel cons (IMO)",
        "Global perc diff",
        "Average vessel annual fuel cons (NavigaTE)",
        "Average vessel annual fuel cons (IMO)",
        "Vessel perc diff",
    ]
    fuel_consumption_df = pd.DataFrame(columns=columns)
    for vessel_type in vessels:
        fuel_consumption_info = {
            "Vessel": vessel_type,
            "Global annual fuel cons (NavigaTE)": 0,
            "Global annual fuel cons (IMO)": annual_fuel_consumption_imo[vessel_type],
        }
        vessel_type_number_navigate = 0
        for vessel in vessels[vessel_type]:
            fuel_consumption_info["Global annual fuel cons (NavigaTE)"] += (
                energy_to_fuel_consumption_lsfo(
                    float(
                        all_results_df["Energy Spend (GJ / year)"][
                            all_results_df["Vessel"] == f"{vessel}_{fuel}"
                        ]
                    )
                )
                * vessel_size_number[vessel]
            )
            vessel_type_number_navigate += vessel_size_number[vessel]
            
# find percentage difference between the average fuel consumptions of the NavigaTE model and the IMO data
        fuel_consumption_info["Global perc diff"] = (
            100
            * (
                fuel_consumption_info["Global annual fuel cons (NavigaTE)"]
                - fuel_consumption_info["Global annual fuel cons (IMO)"]
            )
            / fuel_consumption_info["Global annual fuel cons (IMO)"]
        )

        fuel_consumption_info["Average vessel annual fuel cons (NavigaTE)"] = (
            fuel_consumption_info["Global annual fuel cons (NavigaTE)"]
            / vessel_type_number_navigate
        )

        fuel_consumption_info["Average vessel annual fuel cons (IMO)"] = (
            fuel_consumption_info["Global annual fuel cons (IMO)"]
            / vessel_numbers_imo[vessel_type]
        )

        fuel_consumption_info["Vessel perc diff"] = (
            100
            * (
                fuel_consumption_info["Average vessel annual fuel cons (NavigaTE)"]
                - fuel_consumption_info["Average vessel annual fuel cons (IMO)"]
            )
            / fuel_consumption_info["Average vessel annual fuel cons (IMO)"]
        )

        fuel_consumption_row_df = pd.DataFrame([fuel_consumption_info])
        fuel_consumption_df = pd.concat(
            [fuel_consumption_df, fuel_consumption_row_df], ignore_index=True
        )

    return fuel_consumption_df


# Compare annual fuel consumption rate (tonnes / cargo tonne-mile) evaluated for each vessel type between NavigaTE and IMO
def compare_fuel_consumption_rate(all_results_df, fuel="lsfo"):
    # Total HFO-equivalent fuel consumption for each vessel type in thousand tonnes (from IMO 4th GHG study, Figure 5)
    annual_fuel_consumption_imo = {"bulk": 54359, "container": 63906, "tanker": 74674}

    # Annual cargo miles traveled by vessel class in 2018 (million tonne-miles), from IMO 4th GHG report using option 2 (categorizing trips by journey) (Table 69)
    annual_cargo_miles_imo_opt2 = {"bulk": 26234, "container": 13406, "tanker": 20185}

    columns = [
        "Vessel",
        "Fuel cons rate (NavigaTE)",
        "Fuel cons rate (IMO)",
        "Perc diff",
    ]
    fuel_consumption_rate_df = pd.DataFrame(columns=columns)
    for vessel_type in vessels:
        fuel_consumption_rate_info = {
            "Vessel": vessel_type,
            "Fuel cons rate (NavigaTE)": 0,
            "Fuel cons rate (IMO)": (
                annual_fuel_consumption_imo[vessel_type]
                / annual_cargo_miles_imo_opt2[vessel_type]
            ),  # tonnes of fuel / kilotonne-mile
        }
        vessel_type_number_navigate = 0
        for vessel in vessels[vessel_type]:
            fuel_consumption_rate_info["Fuel cons rate (NavigaTE)"] += (
                energy_to_fuel_consumption_lsfo(
                    float(
                        all_results_df["Energy Spend (GJ / year)"][
                            all_results_df["Vessel"] == f"{vessel}_{fuel}"
                        ].iloc[0]
                    )
                )
                / float(
                    all_results_df["Cargo tonne-miles / year"][
                        all_results_df["Vessel"] == f"{vessel}_{fuel}"
                    ].iloc[0]
                )
                * vessel_size_number[vessel]
                * TONNES_PER_KT
                * TONNES_PER_KT
            )
            vessel_type_number_navigate += vessel_size_number[vessel]

        fuel_consumption_rate_info["Fuel cons rate (NavigaTE)"] = (
            fuel_consumption_rate_info["Fuel cons rate (NavigaTE)"]
            / vessel_type_number_navigate
        )
        fuel_consumption_rate_info["Perc diff"] = (
            100
            * (
                fuel_consumption_rate_info["Fuel cons rate (NavigaTE)"]
                - fuel_consumption_rate_info["Fuel cons rate (IMO)"]
            )
            / fuel_consumption_rate_info["Fuel cons rate (IMO)"]
        )

        fuel_consumption_rate_row_df = pd.DataFrame([fuel_consumption_rate_info])
        fuel_consumption_rate_df = pd.concat(
            [fuel_consumption_rate_df, fuel_consumption_rate_row_df], ignore_index=True
        )

    return fuel_consumption_rate_df


# Compare tank-to-wake emissions evaluated for each vessel type between NavigaTE and IMO
def compare_ttw_emissions(all_results_df, fuel="lsfo"):
    # Annual tank-to-wake CO2 emissions rates (g CO2 / ton-nm) in 2018 (from IMO 4th GHG study, Figure 17)
    annual_ttw_emissions_imo_opt1 = {
        "bulk": 7.3,
        "container": 15.3,
        "tanker": 9.7,  # Using the value for oil tanker
    }

    annual_ttw_emissions_imo_opt2 = {
        "bulk": 6.9,
        "container": 14.8,
        "tanker": 8.2,  # Using the value for oil tanker
    }

    columns = [
        "Vessel",
        "TTW Emission Rate (NavigaTE)",
        "TTW Emission Rate (IMO Opt 1)",
        "Perc diff (IMO Opt 1)",
        "TTW Emission Rate (IMO Opt 2)",
        "Perc diff (IMO Opt 2)",
    ]
    ttw_emission_rate_df = pd.DataFrame(columns=columns)
    for vessel_type in vessels:
        ttw_emission_rate_info = {
            "Vessel": vessel_type,
            "TTW Emission Rate (NavigaTE)": 0,
            "TTW Emission Rate (IMO Opt 1)": annual_ttw_emissions_imo_opt1[vessel_type],
            "TTW Emission Rate (IMO Opt 2)": annual_ttw_emissions_imo_opt2[vessel_type],
        }
        vessel_type_number_navigate = 0
        for vessel in vessels[vessel_type]:
            ttw_emission_rate_info["TTW Emission Rate (NavigaTE)"] += (
                (
                    float(
                        all_results_df["TTW Emissions (tonnes CO2 / year)"][
                            all_results_df["Vessel"] == f"{vessel}_{fuel}"
                        ].iloc[0]
                    )
                )
                / (
                    float(
                        all_results_df["Cargo tonne-miles / year"][
                            all_results_df["Vessel"] == f"{vessel}_{fuel}"
                        ].iloc[0]
                    )
                )
                * KG_PER_TONNE
                * G_PER_KG
                * vessel_size_number[vessel]
            )  # Need to weight the sum by the number of vessels of each size class
            vessel_type_number_navigate += vessel_size_number[vessel]

        # Divide by the total number of vessels of the given type to get the average TTW emission rate
        ttw_emission_rate_info["TTW Emission Rate (NavigaTE)"] = (
            ttw_emission_rate_info["TTW Emission Rate (NavigaTE)"]
            / vessel_type_number_navigate
        )

        ttw_emission_rate_info["Perc diff (IMO Opt 1)"] = (
            100
            * (
                ttw_emission_rate_info["TTW Emission Rate (IMO Opt 1)"]
                - ttw_emission_rate_info["TTW Emission Rate (NavigaTE)"]
            )
            / ttw_emission_rate_info["TTW Emission Rate (IMO Opt 1)"]
        )

        ttw_emission_rate_info["Perc diff (IMO Opt 2)"] = (
            100
            * (
                ttw_emission_rate_info["TTW Emission Rate (IMO Opt 2)"]
                - ttw_emission_rate_info["TTW Emission Rate (NavigaTE)"]
            )
            / ttw_emission_rate_info["TTW Emission Rate (IMO Opt 2)"]
        )

        ttw_emission_rate_row_df = pd.DataFrame([ttw_emission_rate_info])

        ttw_emission_rate_df = pd.concat(
            [ttw_emission_rate_df, ttw_emission_rate_row_df], ignore_index=True
        )

    return ttw_emission_rate_df


# Compare anjnual miles traveled between NavigaTE and IMO
def compare_annual_miles(all_results_df, fuel="lsfo"):
    # Annual miles traveled by vessel size, from IMO 4th GHG report (Table 35)
    annual_miles_imo_dict = {
        "bulk_carrier_capesize_ice": 73223,
        "bulk_carrier_handy_ice": 49094,
        "bulk_carrier_panamax_ice": 59118,
        "container_15000_teu_ice": 98136,
        "container_8000_teu_ice": 100050,
        "container_3500_teu_ice": 87456,
        "tanker_100k_dwt_ice": 53429,
        "tanker_300k_dwt_ice": 72529,
        "tanker_35k_dwt_ice": 45492,
    }

    columns = [
        "Vessel",
        "Annual miles traveled (NavigaTE)",
        "Annual miles traveled (IMO)",
        "Perc diff",
    ]
    miles_traveled_df = pd.DataFrame(columns=columns)
    for vessel_type in vessels:
        for vessel in vessels[vessel_type]:
            annual_miles_navigate = float(
                all_results_df["Miles / year"][
                    all_results_df["Vessel"] == f"{vessel}_{fuel}"
                ]
            )
            annual_miles_imo = annual_miles_imo_dict[vessel]
            perc_diff = (
                100 * (annual_miles_navigate - annual_miles_imo) / annual_miles_imo
            )
            miles_traveled_info = {
                "Vessel": vessel,
                "Annual miles traveled (NavigaTE)": annual_miles_navigate,
                "Annual miles traveled (IMO)": annual_miles_imo,
                "Perc diff": perc_diff,
            }

            miles_traveled_row_df = pd.DataFrame([miles_traveled_info])
            miles_traveled_df = pd.concat(
                [miles_traveled_df, miles_traveled_row_df], ignore_index=True
            )

    return miles_traveled_df


# Compare anjnual miles traveled between NavigaTE, UNCTAD and IMO
def compare_annual_cargo_miles(all_results_df, fuel="lsfo"):
    # Annual cargo miles traveled by vessel class in 2018 (million tonne-miles), from IMO 4th GHG report using option 1 (categorizing trips by vessel) (Table 69)
    annual_cargo_miles_imo_opt1 = {"bulk": 28464, "container": 15153, "tanker": 21817}

    # Annual cargo miles traveled by vessel class in 2018 (million tonne-miles), from IMO 4th GHG report using option 2 (categorizing trips by journey) (Table 69)
    annual_cargo_miles_imo_opt2 = {"bulk": 26234, "container": 13406, "tanker": 20185}

    # Annual cargo miles traveled by vessel class in 2018 (million tonne-miles), from UNCTAD (as reported in Table 69 of IMO 4th GHG report)
    annual_cargo_miles_unctad = {"bulk": 34193, "container": 9535, "tanker": 16686}

    columns = [
        "Vessel",
        "Annual cargo miles (NavigaTE)",
        "Annual cargo miles (IMO opt 1)",
        "Perc diff (IMO opt 1)",
        "Annual cargo miles (IMO opt 2)",
        "Perc diff (IMO opt 2)",
        "Annual cargo miles (UNCTAD)",
        "Perc diff (UNCTAD)",
    ]

    cargo_miles_df = pd.DataFrame(columns=columns)
    for vessel_type in vessels:
        cargo_miles_info = {
            "Vessel": vessel_type,
            "Annual cargo miles (NavigaTE)": 0,
            "Annual cargo miles (IMO opt 1)": annual_cargo_miles_imo_opt1[vessel_type],
            "Annual cargo miles (IMO opt 2)": annual_cargo_miles_imo_opt2[vessel_type],
            "Annual cargo miles (UNCTAD)": annual_cargo_miles_unctad[vessel_type],
        }
        for vessel in vessels[vessel_type]:
            cargo_miles_info["Annual cargo miles (NavigaTE)"] += (
                float(
                    all_results_df["Cargo tonne-miles / year"][
                        all_results_df["Vessel"] == f"{vessel}_{fuel}"
                    ].iloc[0]
                )
                / 1e6
            )

        cargo_miles_info["Perc diff (IMO opt 1)"] = (
            100
            * (
                cargo_miles_info["Annual cargo miles (NavigaTE)"]
                - cargo_miles_info["Annual cargo miles (IMO opt 1)"]
            )
            / cargo_miles_info["Annual cargo miles (IMO opt 1)"]
        )
        cargo_miles_info["Perc diff (IMO opt 2)"] = (
            100
            * (
                cargo_miles_info["Annual cargo miles (NavigaTE)"]
                - cargo_miles_info["Annual cargo miles (IMO opt 2)"]
            )
            / cargo_miles_info["Annual cargo miles (IMO opt 2)"]
        )
        cargo_miles_info["Perc diff (UNCTAD)"] = (
            100
            * (
                cargo_miles_info["Annual cargo miles (NavigaTE)"]
                - cargo_miles_info["Annual cargo miles (UNCTAD)"]
            )
            / cargo_miles_info["Annual cargo miles (UNCTAD)"]
        )

        cargo_miles_row_df = pd.DataFrame([cargo_miles_info])
        cargo_miles_df = pd.concat(
            [cargo_miles_df, cargo_miles_row_df], ignore_index=True
        )

    return cargo_miles_df


def main():
    top_dir = get_top_dir()

    all_results_df = read_results(
        f"{top_dir}/all_outputs_full_fleet/lsfo-1_excel_report.xlsx", "lsfo"
    )

    fuel_consumption_df = compare_fuel_consumption(all_results_df)
    print(fuel_consumption_df)

    fuel_consumption_rate_df = compare_fuel_consumption_rate(all_results_df)
    print(fuel_consumption_rate_df)

    miles_traveled_df = compare_annual_miles(all_results_df)
    print(miles_traveled_df)

    cargo_miles_df = compare_annual_cargo_miles(all_results_df)
    print(cargo_miles_df)

    ttw_emission_rate_df = compare_ttw_emissions(all_results_df)
    print(ttw_emission_rate_df)


main()
