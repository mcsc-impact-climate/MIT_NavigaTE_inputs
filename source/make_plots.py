"""
Date: May 20, 2024
Author: danikam
Purpose: Reads and plots output data from single fuel single vessel NavigaTE runs to compare lifecycle costs, emissions and energy requirements
"""

import numpy as np
import pandas as pd

from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)

# Make lists of all NavigaTE run options
fuels = ['ammonia', 'hydrogen']

blue_pathways = ['SMR_CCS', 'ATR_CCS_R', 'ATR_CCS_OC', 'ATR_CCS_R_OC']
blue_countries = ['USA']

grey_pathways = ['SMR']
grey_countries = ['USA']

electro_pathways = ['grid', 'wind']
electro_countries = ['USA', 'Australia', 'Singapore', 'China']

def get_top_dir():
    '''
    Gets the path to the top level of the git repo (one level up from the source directory)
        
    Parameters
    ----------
    None

    Returns
    -------
    top_dir (string): Path to the top level of the git repo
        
    NOTE: None
    '''
    source_path = Path(__file__).resolve()
    source_dir = source_path.parent
    top_dir = os.path.dirname(source_dir)
    return top_dir
    
def read_results(fuel, pathway, country, filename, all_results_df):
    if fuel=='lsfo':
        results_df_columns = ['Date', 'Time (days)', 'TotalEquivalentWTT', 'TotalEquivalentTTW', 'TotalEquivalentWTW', 'Miles', 'TotalCAPEX', 'TotalExcludingFuelOPEX', 'TotalFuelOPEX', 'ConsumedEnergy_lsfo']
    else:
        results_df_columns = ['Date', 'Time (days)', 'TotalEquivalentWTT', 'TotalEquivalentTTW', 'TotalEquivalentWTW', 'Miles', 'TotalCAPEX', 'TotalExcludingFuelOPEX', 'TotalFuelOPEX', 'ConsumedEnergy_ammonia', 'ConsumedEnergy_lsfo']
    
    results = pd.ExcelFile(filename)
    results_df = pd.read_excel(results, 'Vessels', skiprows=[0,1,2])
    
    results_df.columns = results_df_columns
    results_df = results_df.set_index('Date')
    
    #print(results_df)
    
    results_dict = {}
    results_dict['Fuel'] = fuel
    results_dict['Pathway'] = pathway
    results_dict['Country'] = country
    results_dict['WTT Emissions (tonnes CO2 / year)'] = results_df['TotalEquivalentWTT'].loc['2024-01-01']
    results_dict['TTW Emissions (tonnes CO2 / year)'] = results_df['TotalEquivalentTTW'].loc['2024-01-01']
    results_dict['WTW Emissions (tonnes CO2 / year)'] = results_df['TotalEquivalentWTW'].loc['2024-01-01']
    results_dict['CAPEX (USD / year)'] = results_df['TotalCAPEX'].loc['2024-01-01']
    results_dict['Fuel Cost (USD / year)'] = results_df['TotalFuelOPEX'].loc['2024-01-01']
    results_dict['Other OPEX (USD / year)'] = results_df['TotalExcludingFuelOPEX'].loc['2024-01-01']
    results_dict['Total Cost (USD / year)'] = results_df['TotalCAPEX'].loc['2024-01-01'] + results_df['TotalFuelOPEX'].loc['2024-01-01'] + results_df['TotalExcludingFuelOPEX'].loc['2024-01-01']
    
    if fuel=='lsfo':
        results_dict['Energy Consumed (GJ / year)'] = results_df['ConsumedEnergy_lsfo'].loc['2024-01-01']
    else:
        results_dict['Energy Consumed (GJ / year)'] = results_df['ConsumedEnergy_ammonia'].loc['2024-01-01'] + results_df['ConsumedEnergy_lsfo'].loc['2024-01-01']
    
    results_row_df = pd.DataFrame([results_dict])
    all_results_df = pd.concat([all_results_df, results_row_df], ignore_index=True)
    
    return all_results_df
    
def collect_all_results(top_dir):

    # Collect all data of interest in a dataframe
    columns = ['Fuel', 'Pathway', 'Country', 'WTT Emissions (tonnes CO2 / year)', 'TTW Emissions (tonnes CO2 / year)', 'WTW Emissions (tonnes CO2 / year)', 'CAPEX (USD / year)', 'Fuel Cost (USD / year)', 'Other OPEX (USD / year)', 'Total Cost (USD / year)', 'Energy Consumed (GJ / year)']
    
    all_results_df = pd.DataFrame(columns=columns)
    
    results_filename = f'{top_dir}/all_outputs/report_lsfo.xlsx'
    all_results_df = read_results('lsfo', 'fossil', 'Global', results_filename, all_results_df)

    for fuel in fuels:
        for blue_pathway in blue_pathways:
            for blue_country in blue_countries:
                # Read in the excel file with output results and add them to the dataframe
                results_filename = f'{top_dir}/all_outputs/report_{fuel}_blue_{blue_pathway}_{blue_country}.xlsx'
                all_results_df = read_results(fuel, blue_pathway, blue_country, results_filename, all_results_df)
                
        for grey_pathway in grey_pathways:
            for grey_country in grey_countries:
                # Read in the excel file with output results and add them to the dataframe
                results_filename = f'{top_dir}/all_outputs/report_{fuel}_grey_{grey_pathway}_{grey_country}.xlsx'
                all_results_df = read_results(fuel, grey_pathway, grey_country, results_filename, all_results_df)
                
        for electro_pathway in electro_pathways:
            for electro_country in electro_countries:
                # Read in the excel file with output results and add them to the dataframe
                results_filename = f'{top_dir}/all_outputs/report_{fuel}_electro_{electro_pathway}_{electro_country}.xlsx'
                all_results_df = read_results(fuel, electro_pathway, electro_country, results_filename, all_results_df)
                
    return all_results_df
    
def add_averages(all_results_df):

    # Convert CAPEX and Other OPEX columns to float
    all_results_df['CAPEX (USD / year)'] = pd.to_numeric(all_results_df['CAPEX (USD / year)'], errors='coerce')
    all_results_df['Other OPEX (USD / year)'] = pd.to_numeric(all_results_df['Other OPEX (USD / year)'], errors='coerce')

    # Identify numeric columns for mean calculation
    numeric_cols = all_results_df.select_dtypes(include=[np.number]).columns.tolist()

    # Group by 'Fuel' and 'Pathway', and calculate the mean only for numeric columns
    average_rows = all_results_df.groupby(['Fuel', 'Pathway'])[numeric_cols].mean().reset_index()
    
    # Add a new column to indicate these are averages
    average_rows['Country'] = 'Country Average'
    
    # Append the average_rows back to the original DataFrame
    all_results_df = pd.concat([all_results_df, average_rows], ignore_index=True)
    
    return all_results_df
                
    
def plot_emissions(all_results_df):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    fuel_pathways = []
    emissions_average = {
        'Tank-to-wake': [],
        'Well-to-tank': [],
    }
    emissions_WTW = {}
    
    emissions_WTW['lsfo (fossil)'] = {}
    
    fuel_pathways.append('lsfo (fossil)')
    
    emissions_average['Tank-to-wake'].append(float(all_results_df['TTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']=='lsfo') & (all_results_df['Pathway']=='fossil') & (all_results_df['Country']=='Global')]))

    emissions_average['Well-to-tank'].append(float(all_results_df['WTT Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']=='lsfo') & (all_results_df['Pathway']=='fossil') & (all_results_df['Country']=='Global')]))
    
    emissions_WTW['lsfo (fossil)']['Global'] = float(all_results_df['WTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']=='lsfo') & (all_results_df['Pathway']=='fossil') & (all_results_df['Country']=='Global')])
    
    ax.axvline(emissions_WTW['lsfo (fossil)']['Global'], color='black', ls='--')

    for fuel in fuels:
        for blue_pathway in blue_pathways:
            emissions_WTW[f'{fuel} ({blue_pathway})'] = {}
            fuel_pathways.append(f'{fuel} ({blue_pathway})')
            
            emissions_average['Tank-to-wake'].append(float(all_results_df['TTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==blue_pathway) & (all_results_df['Country']=='Country Average')]))

            emissions_average['Well-to-tank'].append(float(all_results_df['WTT Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==blue_pathway) & (all_results_df['Country']=='Country Average')]))
            
            for blue_country in blue_countries:
                emissions_WTW[f'{fuel} ({blue_pathway})'][blue_country] = float(all_results_df['WTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==blue_pathway) & (all_results_df['Country']==blue_country)])
                
        for grey_pathway in grey_pathways:
            emissions_WTW[f'{fuel} ({grey_pathway})'] = {}
            fuel_pathways.append(f'{fuel} ({grey_pathway})')
            
            emissions_average['Tank-to-wake'].append(float(all_results_df['TTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==grey_pathway) & (all_results_df['Country']=='Country Average')]))

            emissions_average['Well-to-tank'].append(float(all_results_df['WTT Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==grey_pathway) & (all_results_df['Country']=='Country Average')]))
            
            for grey_country in grey_countries:
                emissions_WTW[f'{fuel} ({grey_pathway})'][grey_country] = float(all_results_df['WTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==grey_pathway) & (all_results_df['Country']==grey_country)])
                
        for electro_pathway in electro_pathways:
            emissions_WTW[f'{fuel} ({electro_pathway})'] = {}
            fuel_pathways.append(f'{fuel} ({electro_pathway})')
            
            emissions_average['Tank-to-wake'].append(float(all_results_df['TTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==electro_pathway) & (all_results_df['Country']=='Country Average')]))

            emissions_average['Well-to-tank'].append(float(all_results_df['WTT Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==electro_pathway) & (all_results_df['Country']=='Country Average')]))
            
            for electro_country in electro_countries:
                emissions_WTW[f'{fuel} ({electro_pathway})'][electro_country] = float(all_results_df['WTW Emissions (tonnes CO2 / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==electro_pathway) & (all_results_df['Country']==electro_country)])
    
    bottom = np.zeros(len(fuel_pathways))
    width = 0.5
    
    fuel_pathways_label = []
    for fuel_pathway in fuel_pathways:
        fuel_pathways_label.append(fuel_pathway.replace('_', '-'))
    for emissions_type, emissions in emissions_average.items():
        p = ax.barh(fuel_pathways_label, emissions, width, label=emissions_type, left=bottom)
        bottom += emissions
        
            
    # Overlay WTW for individual countries if there's significant spread between countries
    colours = ['cyan', 'magenta', 'blue', 'red']
    countries_labelled=[]
    for pathway, countries in emissions_WTW.items():
        # Get the index of the pathway from the fuel_pathways list
        index = fuel_pathways.index(pathway)
        
        WTWs = np.asarray(list(emissions_WTW[pathway].values()))
        if np.std(WTWs) / np.mean(WTWs) > 0.01:
            # Plot each value in values at the corresponding index with some offset
            i_country=0
            for country in countries:
                if country in countries_labelled:
                    ax.scatter(emissions_WTW[pathway][country], index, s=50, color=colours[i_country], marker='D')
                else:
                    ax.scatter(emissions_WTW[pathway][country], index, s=50, color=colours[i_country], marker='D', label=country + ' Well-to-wake')
                    countries_labelled.append(country)
                i_country += 1
    
    ax.legend(fontsize=16)
    ax.set_xlabel('CO$_2$e Emissions (tonnes / year)', fontsize=22)
    
    plt.tight_layout()
    plt.savefig('plots/all_emissions.png', dpi=300)
    
def plot_costs(all_results_df):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    fuel_pathways = []
    costs_average = {
        'CAPEX': [],
        'Other OPEX': [],
        'Fuel Cost': [],
    }
    costs_total = {}
    
    costs_total['lsfo (fossil)'] = {}
    
    fuel_pathways.append('lsfo (fossil)')
    
    costs_average['CAPEX'].append(float(all_results_df['CAPEX (USD / year)'][(all_results_df['Fuel']=='lsfo') & (all_results_df['Pathway']=='fossil') & (all_results_df['Country']=='Global')]))

    costs_average['Other OPEX'].append(float(all_results_df['Other OPEX (USD / year)'][(all_results_df['Fuel']=='lsfo') & (all_results_df['Pathway']=='fossil') & (all_results_df['Country']=='Global')]))
    
    costs_average['Fuel Cost'].append(float(all_results_df['Fuel Cost (USD / year)'][(all_results_df['Fuel']=='lsfo') & (all_results_df['Pathway']=='fossil') & (all_results_df['Country']=='Global')]))
    
    costs_total['lsfo (fossil)']['Global'] = float(all_results_df['Total Cost (USD / year)'][(all_results_df['Fuel']=='lsfo') & (all_results_df['Pathway']=='fossil') & (all_results_df['Country']=='Global')])
    
    ax.axvline(costs_total['lsfo (fossil)']['Global'], color='black', ls='--')

    for fuel in fuels:
        for blue_pathway in blue_pathways:
            costs_total[f'{fuel} ({blue_pathway})'] = {}
            fuel_pathways.append(f'{fuel} ({blue_pathway})')
            
            costs_average['CAPEX'].append(float(all_results_df['CAPEX (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==blue_pathway) & (all_results_df['Country']=='Country Average')]))

            costs_average['Other OPEX'].append(float(all_results_df['Other OPEX (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==blue_pathway) & (all_results_df['Country']=='Country Average')]))
            
            costs_average['Fuel Cost'].append(float(all_results_df['Fuel Cost (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==blue_pathway) & (all_results_df['Country']=='Country Average')]))
            
            for blue_country in blue_countries:
                costs_total[f'{fuel} ({blue_pathway})'][blue_country] = float(all_results_df['Total Cost (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==blue_pathway) & (all_results_df['Country']==blue_country)])
                
        for grey_pathway in grey_pathways:
            costs_total[f'{fuel} ({grey_pathway})'] = {}
            fuel_pathways.append(f'{fuel} ({grey_pathway})')
            
            costs_average['CAPEX'].append(float(all_results_df['CAPEX (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==grey_pathway) & (all_results_df['Country']=='Country Average')]))

            costs_average['Other OPEX'].append(float(all_results_df['Other OPEX (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==grey_pathway) & (all_results_df['Country']=='Country Average')]))
            
            costs_average['Fuel Cost'].append(float(all_results_df['Fuel Cost (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==grey_pathway) & (all_results_df['Country']=='Country Average')]))
            
            for grey_country in grey_countries:
                costs_total[f'{fuel} ({grey_pathway})'][grey_country] = float(all_results_df['Total Cost (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==grey_pathway) & (all_results_df['Country']==grey_country)])
                
        for electro_pathway in electro_pathways:
            costs_total[f'{fuel} ({electro_pathway})'] = {}
            fuel_pathways.append(f'{fuel} ({electro_pathway})')
            
            costs_average['CAPEX'].append(float(all_results_df['CAPEX (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==electro_pathway) & (all_results_df['Country']=='Country Average')]))

            costs_average['Other OPEX'].append(float(all_results_df['Other OPEX (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==electro_pathway) & (all_results_df['Country']=='Country Average')]))
            
            costs_average['Fuel Cost'].append(float(all_results_df['Fuel Cost (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==electro_pathway) & (all_results_df['Country']=='Country Average')]))
            
            for electro_country in electro_countries:
                costs_total[f'{fuel} ({electro_pathway})'][electro_country] = float(all_results_df['Total Cost (USD / year)'][(all_results_df['Fuel']==fuel) & (all_results_df['Pathway']==electro_pathway) & (all_results_df['Country']==electro_country)])
    
    bottom = np.zeros(len(fuel_pathways))
    width = 0.5
    
    fuel_pathways_label = []
    for fuel_pathway in fuel_pathways:
        fuel_pathways_label.append(fuel_pathway.replace('_', '-'))
    for costs_type, costs in costs_average.items():
        p = ax.barh(fuel_pathways_label, costs, width, label=costs_type, left=bottom)
        bottom += costs
        
            
    # Overlay WTW for individual countries if there's significant spread between countries
    colours = ['cyan', 'magenta', 'blue', 'red']
    countries_labelled=[]
    for pathway, countries in costs_total.items():
        # Get the index of the pathway from the fuel_pathways list
        index = fuel_pathways.index(pathway)
        
        totals = np.asarray(list(costs_total[pathway].values()))
        if np.std(totals) / np.mean(totals) > 0.01:
            # Plot each value in values at the corresponding index with some offset
            i_country=0
            for country in countries:
                if country in countries_labelled:
                    ax.scatter(costs_total[pathway][country], index, s=50, color=colours[i_country], marker='D')
                else:
                    ax.scatter(costs_total[pathway][country], index, s=50, color=colours[i_country], marker='D', label=country + ' Total')
                    countries_labelled.append(country)
                i_country += 1
    
    ax.legend(fontsize=16)
    ax.set_xlabel('Cost (USD / year)', fontsize=22)
    
    plt.tight_layout()
    plt.savefig('plots/all_costs.png', dpi=300)
    
    
def main():

    # Get the path to the top level of the Git repo
    top_dir = get_top_dir()

    all_results_df = collect_all_results(top_dir)
        
    all_results_df = add_averages(all_results_df)
    
    print(all_results_df)
    
    plot_emissions(all_results_df)
    
    plot_costs(all_results_df)

main()
