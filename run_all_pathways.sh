#!/bin/bash

# Date: June 3, 2024. Updated Oct 2 2024 to reflect updated include file structure and parallelize execution
# Updated May 29, 2025 to support optional country filtering
# Purpose: Run NavigaTE with single pathway and all bulk, container and tanker vessels for all modelled fuel production pathways
# Author: danikam

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")

# Define the source and destination directories relative to the script location
SOURCE_DIR="${SCRIPT_DIR}/includes_global/all_costs_emissions"
DEST_DIR="${SCRIPT_DIR}/includes_global"

mkdir -p "${SCRIPT_DIR}/all_outputs_full_fleet"

# Remove all previously-created .nav files
rm ${SCRIPT_DIR}/single_pathway_full_fleet/*/navs/*.nav

# Create the input .inc files
python3 ${SCRIPT_DIR}/source/make_cost_emissions_files.py

# Clear out any existing excel and log files
rm ${SCRIPT_DIR}/all_outputs_full_fleet/*.xlsx
rm ${SCRIPT_DIR}/Logs/*.log

echo 'Processing lsfo pathway'
navigate --suppress-plots ${SCRIPT_DIR}/single_pathway_full_fleet/lsfo/lsfo.nav
mv ${SCRIPT_DIR}/all_outputs_full_fleet/lsfo_excel_report.xlsx ${SCRIPT_DIR}/all_outputs_full_fleet/lsfo-1_excel_report.xlsx

# Function to process a pathway
process_pathway() {
    local inc_file=$1
    local fuel=$2
    local pathway_name=$3
    local log_file="${SCRIPT_DIR}/Logs/log-${pathway_name}.log"

    echo "Processing ${pathway_name} pathway" >> "$log_file"

    cp "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel}/navs/${pathway_name}.nav" "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel}/${pathway_name}.nav"
    navigate --suppress-plots "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel}/${pathway_name}.nav" >> "$log_file" 2>&1
    rm "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel}/${pathway_name}.nav"
    rm "${SCRIPT_DIR}/single_pathway_full_fleet/${fuel}/${pathway_name}.prt"
}

# Number of parallel processes
N_PARALLEL=16
pids=()

# Capture user-specified countries
COUNTRIES=("$@")

# Get list of files based on countries or all if none specified
if [ ${#COUNTRIES[@]} -eq 0 ]; then
    echo "No countries specified. Processing all .inc files..."
    INC_FILES=("${SOURCE_DIR}"/*.inc)
else
    echo "Filtering for countries: ${COUNTRIES[*]}"
    INC_FILES=()
    for country in "${COUNTRIES[@]}"; do
        for file in "${SOURCE_DIR}"/*"${country}"*.inc; do
            [ -e "$file" ] && INC_FILES+=("$file")
        done
    done
fi

total_files=${#INC_FILES[@]}
processed=0

for inc_file in "${INC_FILES[@]}"; do
    filename=$(basename -- "$inc_file")
    fuel=$(echo $filename | cut -d'-' -f1)
    pathway_name=$(basename -- "$filename" .inc)

    process_pathway "$inc_file" "$fuel" "$pathway_name" &

    pids+=($!)

    if [ "${#pids[@]}" -ge "$N_PARALLEL" ]; then
        for pid in "${pids[@]}"; do
            wait "$pid"
            processed=$((processed+1))
            percentage=$((processed*100/total_files))
            echo "Processed $processed out of $total_files ($percentage%)"
        done
        pids=()
    fi
done

for pid in "${pids[@]}"; do
    wait "$pid"
    processed=$((processed+1))
    percentage=$((processed*100/total_files))
    echo "Processed $processed out of $total_files ($percentage%)"
done

# Convert Excel outputs to CSV
python "${SCRIPT_DIR}/source/convert_excel_files_to_csv.py" --input_dir "${SCRIPT_DIR}/all_outputs_full_fleet" --output_dir "${SCRIPT_DIR}/all_outputs_full_fleet_csv"
