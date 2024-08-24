#!/bin/bash

# Date: June 3, 2024. Updated July 23 to generalize to any .inc input.
# Purpose: Run NavigaTE with single pathway and all bulk, container and tanker vessels for all modelled fuel production pathways
# Author: danikam

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")

# Define the source and destination directories relative to the script location
SOURCE_DIR="${SCRIPT_DIR}/includes_global/all_costs_emissions"
DEST_DIR="${SCRIPT_DIR}/includes_global"

mkdir -p "${SCRIPT_DIR}/all_outputs_full_fleet"

# Create the input .inc files
python ${SCRIPT_DIR}/source/make_cost_emissions_files.py

# Clear out any existing excel files
rm ${SCRIPT_DIR}/all_outputs_full_fleet/*.xlsx

echo 'Processing lsfo pathway'
# Copy the given black pathway into the top level of includes_global and run
navigate --suppress-plots ${SCRIPT_DIR}/single_pathway_full_fleet/lsfo/lsfo.nav
cp ${SCRIPT_DIR}/single_pathway_full_fleet/lsfo/plots/lsfo_excel_report.xlsx ${SCRIPT_DIR}/all_outputs_full_fleet/report-lsfo-1.xlsx

process_pathway() {
    local inc_file=$1
    local fuel=$2
    local pathway_name=$3
    local log_file="${SCRIPT_DIR}/Logs/log-${pathway_name}.log"

    echo "Processing ${pathway_name} pathway" | tee -a "$log_file"

    cp "$inc_file" "${DEST_DIR}/cost_emissions_${fuel}.inc" 2>&1 | tee -a "$log_file"
    navigate --suppress-plots "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel}/${fuel}.nav" 2>&1 | tee -a "$log_file"
    cp "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel}/plots/${fuel}_excel_report.xlsx" "${SCRIPT_DIR}/all_outputs_full_fleet/report-${pathway_name}.xlsx" 2>&1 | tee -a "$log_file"
}

for inc_file in "${SOURCE_DIR}"/*.inc; do

    filename=$(basename -- "$inc_file")
    fuel=$(echo $filename | cut -d'-' -f1)
    pathway_name=$(basename -- "$filename" .inc)
    
    echo $filename $fuel $pathway_name
    process_pathway "$inc_file" "$fuel" "$pathway_name" &
    
done


