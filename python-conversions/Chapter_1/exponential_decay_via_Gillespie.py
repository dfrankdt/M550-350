#!/usr/bin/env python3
"""
Exponential Decay via Gillespie Algorithm and ODE Simulation

This script simulates the exponential decay of particles through a stochastic process
(simulated with the Gillespie algorithm) and compares the results with the solution of an
ODE system describing the evolution of the probability distribution of particle numbers.

Reaction:
    S -> 0 with decay rate alpha

Figures produced:
    Figure 1: A step plot of a single decay trajectory alongside a deterministic decay curve.
    Figure 2: A semilog plot of the same decay trajectory.
    Figure 3: A histogram (as an approximate PDF) of extinction times.
    Figure 4: The evolution of the probability distribution p_k(t) (ODE solution).
    Figure 5: Comparison between the histogram-derived extinction rate and dp₀/dt (ODE prediction).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# For debugging/information: print available styles.
print("Available Matplotlib styles:", plt.style.available)

# Use a valid, visually appealing style from the available list.
# Here we choose "seaborn-v0_8-whitegrid", which gives a white grid background.
plt.style.use("seaborn-v0_8-whitegrid")

# Update rcParams to set moderate and consistent font sizes/line widths.
plt.rcParams.update(
    {
        "axes.titlesize": 16,  # Title size
        "axes.labelsize": 14,  # Axis label size
        "xtick.labelsize": 12,  # Tick label size for x-axis
        "ytick.labelsize": 12,  # Tick label size for y-axis
        "axes.linewidth": 1.2,  # Axis line width
        "lines.linewidth": 2.0,  # Plot line width
    }
)


# =============================================================================
# Helper Function: Right-Hand Side of the ODE System for p_k(t)
# =============================================================================
def de_rhs(t, p, N, alpha):
    """
    Compute the right-hand side of the ODE for the probability distribution p_k(t):

         dp_k/dt = alpha * [(k+1)*p_{k+1} - k*p_k]

    for k = 0, 1, ..., N, with p_N decaying according to dp_N/dt = -alpha * N * p_N.

    Parameters:
        t (float): Time (unused explicitly since the system is autonomous)
        p (ndarray): Array of probabilities for k = 0, 1, ..., N.
        N (int): Initial number of particles.
        alpha (float): Decay rate.

    Returns:
        dpdt (ndarray): Array of time derivatives for each p_k.
    """
    k = np.arange(0, N + 1)
    dpdt = np.zeros(N + 1)
    # For k = 0, 1, ... , N-1: flux from state k+1 minus loss from state k.
    dpdt[:-1] = alpha * (np.arange(1, N + 1) * p[1:] - np.arange(0, N) * p[:-1])
    # For k = N: only decay occurs.
    dpdt[-1] = -alpha * N * p[-1]
    return dpdt


# =============================================================================
# Main Simulation Function
# =============================================================================
def stochastic_decay():
    """
    Simulate exponential particle decay via the Gillespie algorithm and compare it
    with the ODE simulation of the probability distribution p_k(t).

    The reaction is: S -> 0 with rate alpha.

    Generates several figures:
      - Figure 1: A step plot of a single decay trajectory alongside a deterministic decay curve.
      - Figure 2: A semilog plot of the particle number versus time.
      - Figure 3: A histogram (as an approximate PDF) of extinction times.
      - Figure 4: The evolution of the probability distribution p_k(t) (ODE solution).
      - Figure 5: Comparison between the histogram-derived extinction rate and dp₀/dt (ODE prediction).
    """
    # -- Simulation Parameters --
    N = 25  # Initial particle number.
    alpha = 1  # Decay rate.
    K = 10000  # Number of independent trials.

    # Preallocate an array to store reaction times.
    # Rows correspond to decay events (from 0 to N decays),
    # columns correspond to different trials.
    t_arr = np.zeros((N + 1, K))
    # Generate uniform random numbers for the exponential waiting times.
    R = np.random.rand(N, K)

    # ------------------------------
    # Gillespie Simulation Loop
    # ------------------------------
    for j in range(N):
        n = N - j  # Number of particles remaining.
        rxn = alpha * n  # Reaction rate when n particles remain.
        # Sample waiting time from an exponential distribution.
        dt = -np.log(R[j, :]) / rxn
        # Update the decay times for each trial.
        t_arr[j + 1, :] = t_arr[j, :] + dt

    # ------------------------------
    # Plotting: Decay Trajectories
    # ------------------------------
    # Select one trial (here, trial 0) to visualize a decay trajectory.
    trial_index = 0
    decay_times = t_arr[:, trial_index]
    particle_numbers = np.arange(N, -1, -1)  # Particle number from N down to 0.

    # Create a fine time grid for the deterministic decay curve.
    t_fine = np.linspace(0, decay_times[-1], 200)

    # Figure 1: Step plot (stochastic decay) vs. deterministic decay.
    plt.figure(figsize=(8, 6))
    plt.step(decay_times, particle_numbers, where="post", label="Stochastic Decay")
    plt.plot(
        t_fine, N * np.exp(-alpha * t_fine), "--", label="Deterministic: N exp(-αt)"
    )
    plt.xlabel(r"$\alpha t$")
    plt.ylabel("Particle Number")
    plt.title("Decay Trajectory: Stochastic vs. Deterministic")
    plt.legend()
    plt.tight_layout()

    # Figure 2: Semilog plot of the decay trajectory.
    plt.figure(figsize=(8, 6))
    plt.semilogy(decay_times, particle_numbers, label="Stochastic Decay")
    plt.semilogy(t_fine, N * np.exp(-alpha * t_fine), "--", label="Deterministic Decay")
    plt.xlabel(r"$\alpha t$")
    plt.ylabel("Particle Number")
    plt.title("Semilog Plot: Particle Number vs. Time")
    plt.legend()
    plt.tight_layout()

    # ------------------------------
    # Histogram of Extinction Times
    # ------------------------------
    # Extinction time: when particle count reaches 0 (last row of t_arr).
    extinction_times = t_arr[-1, :]
    NN_hist, bins = np.histogram(extinction_times, bins=50, density=False)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    dt_bin = np.mean(np.diff(bins))  # Mean bin width.
    # Convert counts to approximate probability density.
    pdf_data = NN_hist / (K * dt_bin)

    # Compute the sample mean of the extinction times.
    sample_mean = np.mean(extinction_times)
    print("Sample mean extinction time:", sample_mean)
    # Theoretical mean extinction time: sum(1/k) for k = 1 to N.
    theoretical_mean = np.sum(1 / np.arange(1, N + 1))
    print("Theoretical mean extinction time:", theoretical_mean)

    # Figure 3: Histogram (PDF) of extinction times.
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, pdf_data, "*", label="Histogram PDF")
    plt.xlabel(r"$\alpha t$")
    plt.ylabel("Probability Density")
    plt.title("Histogram of Extinction Times")
    plt.legend()
    plt.tight_layout()

    # ------------------------------
    # ODE Simulation for p_k(t)
    # ------------------------------
    # Initial condition: probability 1 at k = N (all particles present).
    p0 = np.zeros(N + 1)
    p0[-1] = 1
    t_end = 12
    tstep = 0.02
    t_eval = np.arange(0, t_end + tstep, tstep)

    # Solve the ODE system using a stiff method (BDF).
    sol = solve_ivp(
        lambda t, p: de_rhs(t, p, N, alpha), [0, t_end], p0, t_eval=t_eval, method="BDF"
    )

    # Figure 4: Evolution of the probability distribution p_k(t).
    plt.figure(figsize=(8, 6))
    for k in range(N + 1):
        plt.plot(sol.t, sol.y[k, :], label=f"$p_{{{k}}}(t)$")
    plt.xlabel(r"$\alpha t$")
    plt.ylabel(r"$p_k(t)$")
    plt.title("Evolution of the Probability Distribution $p_k(t)$")
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    # ------------------------------
    # Comparison: Extinction Rate vs. ODE Prediction
    # ------------------------------
    # For the ODE, dp₀/dt = alpha * p₁(t).
    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, pdf_data, "*", label="Histogram (Extinction Rate)")
    plt.plot(sol.t, alpha * sol.y[1, :], "--", label=r"$\alpha\, p_1(t)$")
    plt.xlabel(r"$\alpha t$")
    plt.ylabel(r"$dp_0/dt$")
    plt.title("Extinction Rate: Data vs. ODE Prediction")
    plt.xticks(np.arange(0, 16, 2))
    plt.xlim([0, 14])
    plt.ylim([0, 0.4])
    plt.legend()
    plt.tight_layout()

    # Display all figures.
    plt.show()


# =============================================================================
# Execute the simulation if the script is run directly.
# =============================================================================
if __name__ == "__main__":
    stochastic_decay()
