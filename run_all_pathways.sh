#!/bin/bash

# Date: June 3, 2024. Updated Oct 2 2024 to reflect updated include file structure and parallelize execution
# Updated May 29, 2025 to support optional keyword filtering
# Updated July 6, 2025 to support optional specification to use input files with original tank and cargo capacities
# Purpose: Run NavigaTE with single pathway and all bulk, container and tanker vessels for all modelled fuel production pathways
# Author: danikam

# Parse command-line arguments
KEYWORDS=()
ORIG_CAPS_FLAG=""
KEEP_XLSX=false
LABEL=""

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -k|--keyword)
      KEYWORDS+=("$2")
      shift 2
      ;;
    -o|--orig_caps)
      ORIG_CAPS_FLAG="--orig_caps"
      shift
      ;;
    -x|--keep_xlsx)
      KEEP_XLSX=true
      shift
      ;;
    -l|--label)
      LABEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")

# Set output directories based on label
OUTPUT_DIR="${SCRIPT_DIR}/all_outputs_full_fleet"
OUTPUT_CSV_DIR="${SCRIPT_DIR}/all_outputs_full_fleet_csv"

if [ -n "$LABEL" ]; then
  OUTPUT_DIR="${SCRIPT_DIR}/all_outputs_full_fleet_${LABEL}"
  OUTPUT_CSV_DIR="${SCRIPT_DIR}/all_outputs_full_fleet_${LABEL}_csv"
  echo "Using label: $LABEL"
fi

# Define the source and destination directories relative to the script location
SOURCE_DIR="${SCRIPT_DIR}/includes_global/all_costs_emissions"
DEST_DIR="${SCRIPT_DIR}/includes_global"

mkdir -p "${OUTPUT_DIR}"

# Remove all previously-created .nav files
rm ${SCRIPT_DIR}/single_pathway_full_fleet/*/navs/*.nav

# Create the input .inc files
python3 ${SCRIPT_DIR}/source/make_cost_emissions_files.py ${ORIG_CAPS_FLAG}

# Clear out any existing excel and log files
if [ "$KEEP_XLSX" = true ]; then
  echo "Keeping previously generated .xlsx files (--keep_xlsx enabled)"
else
  echo "Removing previously generated .xlsx files"
  rm ${OUTPUT_DIR}/*.xlsx
fi
rm ${SCRIPT_DIR}/Logs/*.log

echo 'Processing lsfo pathway'
if [ -n "$ORIG_CAPS_FLAG" ]; then
  navigate --suppress-plots "${SCRIPT_DIR}/single_pathway_full_fleet/lsfo/lsfo_orig_caps.nav"
  OUTPUT_LSFO_FILENAME="lsfo_orig_caps_excel_report.xlsx"
else
  navigate --suppress-plots "${SCRIPT_DIR}/single_pathway_full_fleet/lsfo/lsfo.nav"
  OUTPUT_LSFO_FILENAME="lsfo_excel_report.xlsx"
fi
mv "${SCRIPT_DIR}/all_outputs_full_fleet/${OUTPUT_LSFO_FILENAME}" ${OUTPUT_DIR}/lsfo-1_excel_report.xlsx


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
    if [ -n "$LABEL" ]; then
        mv "${SCRIPT_DIR}/all_outputs_full_fleet/${pathway_name}_excel_report.xlsx" "${OUTPUT_DIR}/${pathway_name}_excel_report.xlsx"
    fi
}

# Number of parallel processes
N_PARALLEL=16
pids=()

# Get list of files based on countries or all if none specified
if [ ${#KEYWORDS[@]} -eq 0 ]; then
    echo "No keywords specified. Processing all .inc files..."
    INC_FILES=("${SOURCE_DIR}"/*.inc)
else
    echo "Filtering for keywords: ${KEYWORDS[*]}"
    INC_FILES=()
    for keyword in "${KEYWORDS[@]}"; do
        for file in "${SOURCE_DIR}"/*"${keyword}"*.inc; do
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
python "${SCRIPT_DIR}/source/convert_excel_files_to_csv.py" --input_dir "${OUTPUT_DIR}" --output_dir "${OUTPUT_CSV_DIR}"

