"""
Date: Jan. 22, 2024
Author: danikam
Purpose: Visualizes the variation of beta*gamma-1 with N for liquid hydrogen
"""

import matplotlib.pyplot as plt
import numpy as np

# Define the range for N
N_limited = np.linspace(0, 500, 1000)

# Calculate the function values for both plots
y_updated = 0.35 / (0.995**N_limited) - 1
ratio_relative_035 = y_updated / y_updated[0]

y_new = 4.9 / (0.995**N_limited) - 1
ratio_relative = y_new / y_new[0]


# Function to create the plots
def create_plot(y_values, ratio_values, ylabel_top, ylabel_bottom, str_save):
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]}
    )

    # Top panel: Original plot
    axs[0].plot(N_limited, y_values, linewidth=2)
    axs[0].set_ylabel(ylabel_top, fontsize=24)
    axs[0].grid(True)
    axs[0].tick_params(axis="both", labelsize=20)

    # Bottom panel: Ratio relative to N=0
    axs[1].plot(N_limited, ratio_values, label="Ratio relative to N=0", linewidth=2)
    axs[1].set_xlabel("N (days)", fontsize=28)
    axs[1].set_ylabel(ylabel_bottom, fontsize=24)
    axs[1].legend(fontsize=24)
    axs[1].grid(True)
    if "mass" in str_save:
        axs[0].axhline(0, color="red", ls="--")
        axs[1].axhline(0, color="red", ls="--")
    axs[1].tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    plt.savefig(f"plots/{str_save}")
    plt.close()


# Plot for 0.35 / 0.995^N - 1
create_plot(
    y_updated,
    ratio_relative_035,
    r"$\beta_\text{mass}\gamma_\text{mass} - 1$",
    "Relative Ratio",
    "beta_gamma_mass.pdf",
)

# Plot for 4.9 / 0.995^N - 1
create_plot(
    y_new,
    ratio_relative,
    r"$\beta_\text{volume}\gamma_\text{volume} - 1$",
    "Relative Ratio",
    "beta_gamma_volume.pdf",
)
