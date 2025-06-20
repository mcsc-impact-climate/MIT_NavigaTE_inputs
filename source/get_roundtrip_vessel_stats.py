"""
Date: June 19, 2025
Author: danikae
Purpose: Calculates the equivalent number of vessels of each class and size needed to complete the observed roundtrips from Singapore to Rotterdam, assuming they 
"""

import json
from common_tools import get_top_dir

DAYS_PER_YEAR = 365.25
HOURS_PER_DAY = 24
top_dir = get_top_dir()

# Load vessel data from JSON file
with open(f'{top_dir}/includes_global/singapore_rotterdam_route_info.json', 'r') as f:
    vessels = json.load(f)

def compute_days_per_voyage(legs):
    total_days = 0
    for leg, data in legs.items():
        distance = data['distance']  # in nautical miles
        speed = data['speed']        # in knots
        sailing_time_days = distance / speed / HOURS_PER_DAY  # convert hours to days
        port_time_days = data['port_duration']
        total_days += sailing_time_days + port_time_days
    return total_days

def compute_total_distance(legs):
    return sum(data['distance'] for data in legs.values())

def compute_stats(vessel):
    days_per_voyage = compute_days_per_voyage(vessel['legs'])
    roundtrips_per_year = DAYS_PER_YEAR / days_per_voyage
    required_vessels = vessel['annual_voyages'] / roundtrips_per_year
    distance_per_voyage = compute_total_distance(vessel['legs'])
    annual_distance = distance_per_voyage * roundtrips_per_year
    return days_per_voyage, roundtrips_per_year, required_vessels, annual_distance

# Print header
print(f"{'Vessel':<25} {'Days/rt':>10} {'RTs/year':>10} {'Vessels':>10} {'Annual dist (nm)':>18}")
print("-" * 75)

# Print results
for vessel in vessels:
    days, rpy, n_vessels, annual_dist = compute_stats(vessel)
    print(f"{vessel['vessel_name']:<25} {days:10.2f} {rpy:10.2f} {n_vessels:10.2f} {annual_dist:18.0f}")

