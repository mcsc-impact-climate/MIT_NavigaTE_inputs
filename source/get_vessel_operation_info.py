"""
Date: May 15, 2025
Purpose: Use default vessel definitions to calculate average vessel speed and cargo capacity utilization for each vessel class and size.
"""

from common_tools import get_top_dir
import re
import numpy as np
import pandas as pd

ROUTE_DIR_NAVIGATE = "NavigaTE/navigate/defaults/installation/Route"

# Vessel type and size information
vessels = {
    "bulk_carrier": [
        "bulk_carrier_capesize",
        "bulk_carrier_handy",
        "bulk_carrier_panamax",
    ],
    "container": [
        "container_15000_teu",
        "container_8000_teu",
        "container_3500_teu",
    ],
    "tanker": ["tanker_100k_dwt", "tanker_300k_dwt", "tanker_35k_dwt"],
    "gas_carrier": ["gas_carrier_100k_cbm"],
}

def get_route_properties(top_dir, vessel_class):
    """
    Fetches relevant route properties for the given vessel type and class.
    
    Parameters
    ----------
    vessel_class : str
        Unique keyword in the filename for the given type and class.

    Returns
    -------
    route_properties_dict : dict
        Dictionary containing the route properties for the given vessel and size.
    """
    
    # Construct the filepath to the vessel .inc file for the given vessel
    filepath = f"{top_dir}/{ROUTE_DIR_NAVIGATE}/globalized_{vessel_class}.inc"
    
    # Initialize dictionary to store the extracted properties
    route_properties_dict = {}
    
    # Define regex patterns for flexible matching
    time_at_sea_pattern = re.compile(r"TimeAtSea\s*=\s*([\d.]+)")
    condition_distribution_pattern = re.compile(r"ConditionDistribution\s*=\s*\[([\d.,\s]+)\]")
    speeds_pattern = re.compile(r"Speeds\s*=\s*\[([\d.,\s]+)\]")
    capacity_utilization_pattern = re.compile(r"CapacityUtilizations\s*=\s*\[([\d.,\s]+)\]")
    
    # Read and parse the file
    try:
        with open(filepath, 'r') as file:
            content = file.readlines()
            
            for line in content:
                # Ignore lines that start with '#' or remove comments within lines
                line = line.split('#')[0].strip()
                if not line:
                    continue
                
                # Extract TimeAtSea
                time_at_sea_match = time_at_sea_pattern.search(line)
                if time_at_sea_match:
                    route_properties_dict['TimeAtSea'] = float(time_at_sea_match.group(1))
                
                # Extract ConditionDistribution
                condition_distribution_match = condition_distribution_pattern.search(line)
                if condition_distribution_match:
                    route_properties_dict['ConditionDistribution'] = np.asarray([
                        float(value.strip()) for value in condition_distribution_match.group(1).split(',')
                    ])
                
                # Extract Speeds
                speeds_match = speeds_pattern.search(line)
                if speeds_match:
                    route_properties_dict['Speeds'] = np.asarray([
                        float(value.strip()) for value in speeds_match.group(1).split(',')
                    ])
                
                # Extract CapacityUtilizations
                capacity_utilization_match = capacity_utilization_pattern.search(line)
                if capacity_utilization_match:
                    route_properties_dict['CapacityUtilizations'] = np.asarray([
                        float(value.strip()) for value in capacity_utilization_match.group(1).split(',')
                    ])
                
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the file: {e}")
        return None
    
    return route_properties_dict
    
def compute_average_route_properties(top_dir, vessels_dict):
    """
    Computes the average speed and utilization for each vessel class.

    Parameters
    ----------
    top_dir : str
        Top-level directory where the route files are stored.
    vessels_dict : dict
        Dictionary mapping vessel types to lists of vessel class identifiers.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing vessel_class, vessel_type, average_speed, and average_utilization.
    """
    results = []

    for vessel_type, vessel_classes in vessels_dict.items():
        for vessel_class in vessel_classes:
            route_properties = get_route_properties(top_dir, vessel_class)
            if route_properties is None:
                continue  # Skip if file could not be read or parsed

            try:
                avg_speed = np.sum(route_properties['Speeds'] * route_properties['ConditionDistribution'])
                avg_utilization = np.sum(route_properties['CapacityUtilizations'] * route_properties['ConditionDistribution'])

                results.append({
                    "vessel_class": vessel_class,
                    "vessel_type": vessel_type,
                    "average_speed": avg_speed,
                    "average_utilization": avg_utilization
                })
            except KeyError as e:
                print(f"Missing expected data for {vessel_class}: {e}")

    return pd.DataFrame(results)


def main():
    top_dir = get_top_dir()
    df = compute_average_route_properties(top_dir, vessels)
    print(df)
    
if __name__ == "__main__":
    main()
