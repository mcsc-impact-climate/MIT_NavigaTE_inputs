"""
Date: June 6, 2025
Author: danikae
Purpose: Calculates TTW emission rates, in ton species / ton fuel based on g / MJ from GREET
"""

from common_tools import read_fuel_labels

G_PER_KG = 1000

fuel_labels = read_fuel_labels()

fuel_labels["TTW CO2 (kg CO2 / kg fuel)"] = (
    fuel_labels["TTW CO2 (g/MJ)"]
    * fuel_labels["Lower Heating Value (MJ / kg)"]
    / G_PER_KG
)

fuel_labels["TTW N2O (kg N2O / kg fuel)"] = (
    fuel_labels["TTW N2O (g/MJ)"]
    * fuel_labels["Lower Heating Value (MJ / kg)"]
    / G_PER_KG
)

fuel_labels["TTW Black Carbon (kg BC / kg fuel)"] = (
    fuel_labels["TTW Black Carbon (g/MJ)"]
    * fuel_labels["Lower Heating Value (MJ / kg)"]
    / G_PER_KG
)

fuel_labels["TTW CH4 (kg CH4 / kg fuel)"] = (
    fuel_labels["TTW CH4 (g/MJ)"]
    * fuel_labels["Lower Heating Value (MJ / kg)"]
    / G_PER_KG
)

for fuel in fuel_labels.index:
    print(f"\n\n======= Fuel: {fuel} =======")
    co2_ttw = fuel_labels["TTW CO2 (kg CO2 / kg fuel)"].loc[fuel]
    n2o_ttw = fuel_labels["TTW N2O (kg N2O / kg fuel)"].loc[fuel]
    bc_ttw = fuel_labels["TTW Black Carbon (kg BC / kg fuel)"].loc[fuel]
    ch4_ttw = fuel_labels["TTW CH4 (kg CH4 / kg fuel)"].loc[fuel]
    print(f"CO2 TTW: {co2_ttw}")
    print(f"N2O TTW: {n2o_ttw}")
    print(f"Black Carbon TTW: {bc_ttw}")
    print(f"CH4: {ch4_ttw}")
