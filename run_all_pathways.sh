#!/bin/bash

# Date: June 3, 2024. Updated July 23 to generalize to any .inc input.
# Purpose: Run NavigaTE with single pathway and all bulk, container and tanker vessels for all modelled fuel production pathways
# Author: danikam

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")

# Define the source and destination directories relative to the script location
SOURCE_DIR="${SCRIPT_DIR}/includes_global/all_costs_emissions"
DEST_DIR="${SCRIPT_DIR}/includes_global"

# Should be at most the number of available CPU cores
MAX_PARALLEL_PROCESSES=8

mkdir -p "${SCRIPT_DIR}/all_outputs_full_fleet"

# Create the input .inc files
python ${SCRIPT_DIR}/source/make_cost_emissions_files.py

# Clear out any existing excel files
rm ${SCRIPT_DIR}/all_outputs_full_fleet/*.xlsx

echo 'Processing lsfo pathway'
# Copy the given black pathway into the top level of includes_global and run
navigate --suppress-plots ${SCRIPT_DIR}/single_pathway_full_fleet/lsfo_pathway/lsfo_pathway.nav
cp ${SCRIPT_DIR}/single_pathway_full_fleet/lsfo_pathway/plots/lsfo_pathway_excel_report.xlsx ${SCRIPT_DIR}/all_outputs_full_fleet/report_lsfo-1.xlsx

process_pathway() {
    local inc_file=$1
    local fuel_type=$2
    local pathway_type=$3
    local pathway_name=$4

    echo "Processing ${pathway_name} pathway"

    cp "$inc_file" "${DEST_DIR}/cost_emissions_${fuel_type}_${pathway_type}.inc"
    echo navigate --suppress-plots "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel_type}_${pathway_type}_pathway/${fuel_type}_${pathway_type}_pathway.nav"
    navigate --suppress-plots "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel_type}_${pathway_type}_pathway/${fuel_type}_${pathway_type}_pathway.nav"
    cp "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel_type}_${pathway_type}_pathway/plots/${fuel_type}_${pathway_type}_pathway_excel_report.xlsx" "${SCRIPT_DIR}/all_outputs_full_fleet/report_${pathway_name}.xlsx"
}

export -f process_pathway
export SCRIPT_DIR DEST_DIR

# Loop through all .inc files in the source directory
for inc_file in "${SOURCE_DIR}"/*.inc; do

    filename=$(basename -- "$inc_file")
    fuel=$(echo $filename | cut -d'-' -f1)
    pathway_type=$(echo $filename | cut -d'-' -f2)
    pathway_name=$(basename -- "$filename" .inc)
    
    echo $filename $fuel $pathway_type $pathway_name

    if [[ $pathway_type == "blue" || $pathway_type == "electro" || $pathway_type == "grey" ]]; then
        process_pathway "$inc_file" "$fuel" "$pathway_type" "$pathway_name"
    fi
done

