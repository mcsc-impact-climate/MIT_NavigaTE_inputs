#!/bin/bash

# Set the max number of parallel jobs
N_PARALLEL=16
if [ "$1" != "" ]; then
  N_PARALLEL="$1"
fi

SCRIPT_PATH="source/calculate_fuel_costs_emissions.py"

# Function to run a single year
run_year() {
    YEAR=$1
    echo "Running year $YEAR..."
    python "$SCRIPT_PATH" -y "$YEAR"
}

# Track background PIDs
JOBS=()

for YEAR in $(seq 2024 2050); do
    run_year "$YEAR" &
    JOBS+=($!)

    # Check if we hit the limit
    while [ "${#JOBS[@]}" -ge "$N_PARALLEL" ]; do
        for i in "${!JOBS[@]}"; do
            if ! kill -0 "${JOBS[i]}" 2>/dev/null; then
                unset 'JOBS[i]'  # Remove finished job
            fi
        done
        sleep 1  # Pause before checking again
    done
done

# Wait for remaining jobs
wait
echo "All years completed."
