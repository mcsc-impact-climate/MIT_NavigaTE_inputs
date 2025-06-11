"""
Date: Jan. 21, 2025
Author: danikam
Purpose: Reads in vessel ranges and plots them as a histogram
"""

import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
file_path = "data/vessel_ranges.csv"  # Update this path if necessary
df = pd.read_csv(file_path)

# Extract the relevant columns
vessels = df["Vessel"]
ranges = df["lsfo"]

# Plot horizontal bars
plt.figure(figsize=(10, 6))
plt.barh(vessels, ranges, color="skyblue", edgecolor="black")

# Add labels and title
plt.xlabel("Range (nautical miles)", fontsize=20)
plt.tick_params(axis="y", labelsize=18)
plt.tick_params(axis="x", labelsize=18)
plt.tight_layout()

# Show the plot
plt.savefig("plots/vessel_ranges.png", dpi=300)
plt.savefig("plots/vessel_ranges.pdf", dpi=300)
