#!/usr/bin/env python3
"""
Plots for the Poisson Process

This script creates a plot of the Poisson distribution for j = 0 to 4 over the time interval t ∈ [0, 10].
The Poisson probability function is given by

    p_j(t) = (t^j * exp(-t)) / j!

for each j = 0, 1, 2, 3, 4. The plot shows:
    - Curves for p_j(t) for j = 0, 1, ..., 4.
    - Text annotations at specific coordinates to label each curve.

If the cmyk flag is set to 1, the figure will be saved as an EPS file (note: matplotlib does not automatically
convert to true CMYK, so further processing might be needed if that is required).
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# -----------------------------------------------------------------------------
# Configure Visual Settings
# -----------------------------------------------------------------------------
# Print available styles for debugging purposes.
print("Available Matplotlib styles:", plt.style.available)

# Use a valid, visually appealing style.
plt.style.use("seaborn-v0_8-whitegrid")

# Update rcParams for consistent and moderate font sizes and line widths.
plt.rcParams.update(
    {
        "axes.titlesize": 16,  # Title font size
        "axes.labelsize": 14,  # Axis label font size
        "xtick.labelsize": 12,  # Tick label size (x-axis)
        "ytick.labelsize": 12,  # Tick label size (y-axis)
        "axes.linewidth": 1.2,  # Axis line width
        "lines.linewidth": 2.0,  # Plot line width
    }
)


# -----------------------------------------------------------------------------
# Function: Plot the Poisson Process Distributions
# -----------------------------------------------------------------------------
def plot_poisson_process(cmykflg=0):
    """
    Plot the Poisson process distribution for j = 0, 1, 2, 3, 4 on the interval t ∈ [0, 10].

    Parameters:
        cmykflg (int): If 1, the plot is saved as an EPS file (for CMYK-compatible output).
                       Default is 0 (do not save, just display).
    """
    # Create a time array from 0 to 10 with a 0.01 step.
    t = np.arange(0, 10.01, 0.01)

    # Create a new figure with a fixed size.
    plt.figure(figsize=(8, 6))

    # Loop over j = 0 to 4 (MATLAB's j=1:5 with j-1 yields j=0,...,4).
    # Compute and plot the Poisson probability function:
    #   p_j(t) = (t^j * exp(-t)) / factorial(j)
    for j in range(5):
        f = np.power(t, j) * np.exp(-t) / math.factorial(j)
        plt.plot(t, f, label=f"$p_{{{j}}}(t)$")

    # Add text annotations at specified coordinates (using font size 20 as in MATLAB).
    plt.text(0.5, 0.8, "j=0", fontsize=20)
    plt.text(1.4, 0.4, "j=1", fontsize=20)
    plt.text(2.2, 0.31, "j=2", fontsize=20)
    plt.text(3.3, 0.27, "j=3", fontsize=20)
    plt.text(6.0, 0.18, "j=4", fontsize=20)

    # Label axes and add a title.
    plt.xlabel(r"$\alpha t$")
    plt.ylabel(r"$p_j(t)$")
    plt.title("Poisson Process Distributions")

    # Add a legend to help identify each curve.
    plt.legend()

    # Ensure nothing overlaps in the figure layout.
    plt.tight_layout()

    # Save the figure if the cmyk flag is enabled.
    if cmykflg == 1:
        # Save as an EPS file. For true CMYK output further conversion might be necessary.
        plt.savefig("../../figs_c/chapt_1/poisson_plots.eps", format="eps")

    # Display the plot.
    plt.show()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Set cmykflg to 0 (or 1 if you wish to save the output file).
    plot_poisson_process(cmykflg=0)
q
