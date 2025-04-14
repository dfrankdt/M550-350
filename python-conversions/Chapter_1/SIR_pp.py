#!/usr/bin/env python3
"""
Phase‐Plane Analysis for the SIR Epidemic Model

This script performs a phase‐plane analysis for the classical SIR model
using a stiff ODE solver. The SIR model is given by:

    dS/dt = -α * S * I
    dI/dt =  α * S * I - β * I
    dR/dt =  β * I

where α and β are positive parameters. The script then:
  - Integrates the system over a fixed time interval for several initial conditions.
  - Plots the phase portrait (S vs. I) for the epidemic for different initial susceptible values.
  - Plots a vertical nullcline (S = β/α).
  - Annotates the phase portrait.
  - Generates a second figure that shows the relationship between
    R₀ = α s(0)/β and s(∞)/s(0) via the formula:
       s(∞)/s(0) = log(s(0))/(s(0)-1)

Adjust the parameters and the save path for the figures as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
# Configure Visual Settings
# -----------------------------------------------------------------------------
# Print available styles (for reference).
print("Available Matplotlib styles:", plt.style.available)

# Use a valid, visually appealing style.
plt.style.use("seaborn-v0_8-whitegrid")

# Update rcParams for consistent and moderate font sizes and line widths.
plt.rcParams.update(
    {
        "axes.titlesize": 16,  # Title font size
        "axes.labelsize": 14,  # Axis label size
        "xtick.labelsize": 12,  # Tick label sizes
        "ytick.labelsize": 12,
        "axes.linewidth": 1.0,  # Axes line width as in MATLAB code
        "lines.linewidth": 1.2,  # Line width for plots
        "patch.linewidth": 0.7,  # Patch line width
    }
)


# -----------------------------------------------------------------------------
# ODE Right-Hand Side: SIR Model
# -----------------------------------------------------------------------------
def sir_rhs(t, s, alp, bet):
    """
    Compute the right-hand side of the SIR model ODE system.

    Parameters:
        t (float): Time (not explicitly used because the system is autonomous).
        s (array-like): State vector [S, I, R].
        alp (float): Parameter α (infection rate coefficient).
        bet (float): Parameter β (recovery rate).

    Returns:
        dsdt (ndarray): Time derivative of the state vector.
    """
    S, I, R = s  # Unpack current state: susceptibles, infectives, removed.
    dSdt = -alp * S * I
    dIdt = alp * S * I - bet * I
    dRdt = bet * I
    return [dSdt, dIdt, dRdt]


# -----------------------------------------------------------------------------
# Phase Plane Analysis Function for the SIR Model
# -----------------------------------------------------------------------------
def SIR_phase_plane():
    """
    Carry out phase-plane analysis for the SIR epidemic model.

    - Set parameters: alp = 1 and bet = 1.
    - Define a time span [0, 50] with a time step of 0.01.
    - Loop over different initial susceptible values in s0_list.
      Each initial condition is defined as [s0, 0.001, 0].
    - Solve the SIR ODE system and plot S (susceptible) vs. I (infective).
    - Plot a vertical dashed line at S = bet/alp (the S-nullcline).
    - Add an annotation arrow to the phase portrait.
    - Create a second figure that plots
            as0byb = log(s0)/(s0 - 1)
      vs.
            sinfbys0 = s0,
      corresponding to the theoretical relationship:
            R₀ = α s(0)/β   and   s(∞)/s(0) = log(s(0))/(s(0)-1)
    """
    # ---------------------------
    # Set SIR model parameters
    # ---------------------------
    alp = 1
    bet = 1
    Imx = 1.0  # Maximum value for infectives (I) on plot

    # ---------------------------
    # Time Discretization for ODE Integration
    # ---------------------------
    tstep = 0.01
    t_end = 50
    t_eval = np.arange(0, t_end + tstep, tstep)

    # ---------------------------
    # Figure 1: Phase Portrait (S vs. I)
    # ---------------------------
    plt.figure(figsize=(8, 6))

    # List of initial susceptible values to simulate.
    s0_list = [1.5, 2.0, 2.5]
    for s0_initial in s0_list:
        # Define initial state: [S0, I0, R0] with I0 small and R0 zero.
        s0 = [s0_initial, 0.001, 0]
        sol = solve_ivp(
            lambda t, s: sir_rhs(t, s, alp, bet),
            [0, t_end],
            s0,
            t_eval=t_eval,
            method="BDF",
        )
        # Plot S (sol.y[0]) vs I (sol.y[1])
        plt.plot(sol.y[0], sol.y[1], label=f"s₀ = {s0_initial}")

    # Label the phase plane.
    plt.xlabel(r"$\alpha s/\beta$")
    plt.ylabel(r"$\alpha i/\beta$")
    plt.title("Phase Plane for the SIR Model")
    plt.axis([0, 3, 0, Imx])

    # Plot the vertical nullcline at S = bet/alp (i.e. S = 1, since alp=bet=1).
    plt.plot([bet / alp, bet / alp], [0, Imx], "--", label="Nullcline S = β/α")

    # Add an annotation arrow (using axes fraction coordinates).
    plt.annotate(
        "", xy=(0.35, 0.68), xytext=(0.42, 0.68), arrowprops=dict(arrowstyle="->", lw=2)
    )

    plt.legend()
    plt.tight_layout()

    # Optionally, save the figure as an EPS file with CMYK color if desired.
    # (Adjust the path as needed.)
    # plt.savefig('../../figs_c/chapt_1/sir_pp.eps', format='eps')

    # ---------------------------
    # Figure 2: Theoretical Relationship Plot
    # ---------------------------
    # Define s0 values (should be in the range (0,1]) for the formula.
    sinfbys0 = np.arange(0.01, 1.01, 0.01)
    # Compute as0byb = log(s0)/(s0 - 1)
    as0byb = np.log(sinfbys0) / (sinfbys0 - 1)

    plt.figure(figsize=(8, 6))
    plt.plot(as0byb, sinfbys0, linewidth=2)
    plt.xlabel(r"$R_0=\alpha s(0)/\beta$")
    plt.ylabel(r"$s(\infty)/s(0)$")
    plt.title("Theoretical Relationship Between $R_0$ and Final Susceptible Fraction")
    plt.tight_layout()

    # Display all figures.
    plt.show()


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    SIR_phase_plane()
