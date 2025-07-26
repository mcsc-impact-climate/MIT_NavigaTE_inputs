"""
Date: Jan. 22, 2024
Author: danikam
Purpose: Visualizes the variation of beta*gamma-1 with N for liquid hydrogen
"""

import matplotlib.pyplot as plt
import numpy as np

# Define the range for N
N = np.linspace(0, 350, 1000)
f_pilot = 0.25
f_port = 0.4
r_f = 0.157125/100

f_boiloffs = (1/N) * (1 / (1 - r_f) ** (f_port * N)) * (1 - (1 - r_f) ** (f_port * N)) / (1 - (1 - r_f)**f_port)

# Calculate the function values for both plots
beta_m = 0.342
beta_V = 4.79

y_m = N * (beta_m * f_boiloffs + f_pilot - 1)
y_V = N * (beta_V * f_boiloffs + f_pilot - 1)


# Function to create the plots
def create_plot(y_values, ylabel, str_save):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Main panel: y_values vs. N
    ax.plot(N, y_values, linewidth=2)
    ax.set_xlabel("N (days)", fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.grid(True)
    ax.tick_params(axis="both", labelsize=20)

    if "mass" in str_save:
        ax.axhline(0, color="red", ls="--")

    plt.tight_layout()
    plt.savefig(f"plots/{str_save}")
    plt.close()


# Plot for mass
create_plot(
    y_m,
    r"$\beta^\text{mass} \times f_f^\text{boiloff} + f^\text{pilot} - 1$",
    "beta_gamma_mass.pdf",
)

# Plot for volume
create_plot(
    y_V,
    r"$\beta^\text{volume} \times f_f^\text{boiloff} + f^\text{pilot} - 1$",
    "beta_gamma_volume.pdf",
)

# Plot f_boiloffs
create_plot(
    f_boiloffs,
    r"$f_\text{LH2}^\text{boiloff}$",
    "boiloff_vs_N.pdf",
)
