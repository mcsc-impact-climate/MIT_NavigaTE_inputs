#!/bin/bash

# Date: May 19, 2024
# Purpose: Run NavigaTE with single vessel and single pathway for all modelled fuel production pathways
# Author: danikam

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")

# Define the source and destination directories relative to the script location
SOURCE_DIR="${SCRIPT_DIR}/includes_global/all_costs_emissions"
DEST_DIR="${SCRIPT_DIR}/includes_global"

FUELS=(ammonia hydrogen)

BLUE_PATHWAYS=(SMR_CCS ATR_CCS_R ATR_CCS_OC ATR_CCS_R_OC)
BLUE_COUNTRIES=(USA)

GREY_PATHWAYS=(SMR)
GREY_COUNTRIES=(USA)

ELECTRO_PATHWAYS=(grid wind)
ELECTRO_COUNTRIES=(USA Australia Singapore China)

# Should be at most the number of available CPU cores
MAX_PARALLEL_PROCESSES=8

mkdir -p "${SCRIPT_DIR}/all_outputs"

echo 'Processing lsfo pathway'
# Copy the given black pathway into the top level of includes_global and run
navigate --suppress-plots ${SCRIPT_DIR}/single_pathway_single_vessel/lsfo_pathway/lsfo_pathway.nav
cp ${SCRIPT_DIR}/single_pathway_single_vessel/lsfo_pathway/plots/lsfo_pathway_excel_report.xlsx ${SCRIPT_DIR}/all_outputs/report_lsfo.xlsx

for fuel in "${FUELS[@]}"; do
    # Loop through blue pathways
    echo 'Processing blue pathways'
    for blue_pathway in "${BLUE_PATHWAYS[@]}"; do
        for blue_country in "${BLUE_COUNTRIES[@]}"; do
            # Copy the given blue pathway into the top level of includes_global and run
            cp "${SOURCE_DIR}/cost_emissions_${fuel}_blue_${blue_pathway}_${blue_country}.inc" "${DEST_DIR}/cost_emissions_${fuel}_blue.inc"
            navigate --suppress-plots ${SCRIPT_DIR}/single_pathway_single_vessel/${fuel}_blue_pathway/${fuel}_blue_pathway.nav
            cp ${SCRIPT_DIR}/single_pathway_single_vessel/${fuel}_blue_pathway/plots/${fuel}_blue_pathway_excel_report.xlsx ${SCRIPT_DIR}/all_outputs/report_${fuel}_blue_${blue_pathway}_${blue_country}.xlsx
        done
    done

    # Loop through electrolytic pathways
    echo 'Processing electrolytic pathways'
    for electro_pathway in "${ELECTRO_PATHWAYS[@]}"; do
        for electro_country in "${ELECTRO_COUNTRIES[@]}"; do
            # Copy the given electro pathway into the top level of includes_global and run
            cp "${SOURCE_DIR}/cost_emissions_${fuel}_electro_${electro_pathway}_${electro_country}.inc" "${DEST_DIR}/cost_emissions_${fuel}_electro.inc"
            navigate --suppress-plots ${SCRIPT_DIR}/single_pathway_single_vessel/${fuel}_electro_pathway/${fuel}_electro_pathway.nav
            cp ${SCRIPT_DIR}/single_pathway_single_vessel/${fuel}_electro_pathway/plots/${fuel}_electro_pathway_excel_report.xlsx ${SCRIPT_DIR}/all_outputs/report_${fuel}_electro_${electro_pathway}_${electro_country}.xlsx
        done
    done

    # Loop through grey pathways
    echo 'Processing grey pathways'
    for grey_pathway in "${GREY_PATHWAYS[@]}"; do
        for grey_country in "${GREY_COUNTRIES[@]}"; do
            # Copy the given grey pathway into the top level of includes_global and run
            cp "${SOURCE_DIR}/cost_emissions_${fuel}_grey_${grey_pathway}_${grey_country}.inc" "${DEST_DIR}/cost_emissions_${fuel}_grey.inc"
            navigate --suppress-plots ${SCRIPT_DIR}/single_pathway_single_vessel/${fuel}_grey_pathway/${fuel}_grey_pathway.nav
            cp ${SCRIPT_DIR}/single_pathway_single_vessel/${fuel}_grey_pathway/plots/${fuel}_grey_pathway_excel_report.xlsx ${SCRIPT_DIR}/all_outputs/report_${fuel}_grey_${grey_pathway}_${grey_country}.xlsx
        done
    done
done
