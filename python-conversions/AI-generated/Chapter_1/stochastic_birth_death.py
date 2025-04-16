#!/usr/bin/env python3
"""
This Python script simulates a birth–death process using the Gillespie (stochastic) algorithm,
and also solves a corresponding system of ordinary differential equations (ODEs)
to obtain extinction probabilities. The code is written with detailed comments to help beginners understand
each step. You can modify the parameters and structure as needed for your own work.

Reactions simulated:
  1. Death:      S -> I      at rate alpha    (reduces the number of individuals S by 1, increases deaths I by 1)
  2. Birth:      S -> 2S     at rate beta     (increases the number of individuals S by 1, I remains unchanged)

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ------------------------------------------------------------
# Define the ODE right-hand side function
# ------------------------------------------------------------
def de_rhs(t, p, Nde, alpha, beta):
    """
    Computes the right-hand side for the system of ODEs for extinction probabilities.

    Parameters:
        t     : Time (not used explicitly since the system is autonomous).
        p     : Current probability vector (length = Nde+1).
        Nde   : Maximum index for the p_k states.
        alpha : Reaction rate for death.
        beta  : Reaction rate for birth.

    Returns:
        dp/dt: A numpy array containing the time derivatives of p (same shape as p).

    The ODE system (in index form, with k = 0,1,...,Nde) is:
      dp[k]/dt = -(alpha*k + beta*k)*p[k]
                 + (if k < Nde: alpha*(k+1)*p[k+1], else 0)
                 + (if k > 0: beta*(k)*p[k-1], else 0)
    """
    # Create an array of indices: [0, 1, 2, ..., Nde]
    K = np.arange(0, Nde + 1)

    # Loss term: each state loses probability at rate (alpha+beta) multiplied by the index (k)
    p_prime = -(alpha * K + beta * K) * p

    # Birth term: For states 0 to Nde-1, probability flows from state k+1 backwards to state k.
    # We use p[1:] (states 1 to Nde) and multiply by alpha*(k+1) because K[1:] is [1,2,...,Nde].
    p_birth = np.zeros_like(p)
    p_birth[:-1] = alpha * (np.arange(1, Nde + 1)) * p[1:]

    # Death term: For states 1 to Nde, probability flows from state k-1 forward to state k.
    # Multiply p[:-1] (states 0 to Nde-1) by beta*k, where k is given by np.arange(0, Nde)
    p_death = np.zeros_like(p)
    p_death[1:] = beta * (np.arange(0, Nde)) * p[:-1]

    # Add the contributions from the birth and death terms to the loss term.
    p_prime += p_birth + p_death
    return p_prime


# ------------------------------------------------------------
# Main simulation function: Gillespie Simulation and ODE solving
# ------------------------------------------------------------
def main():
    # ============================================================
    # Set simulation parameters and initialize state variables
    # ============================================================

    # Flag: if True, figure files are saved in a specific format. (Here we use PNG.)
    cmykflg = True

    # Initial number of individuals (population size)
    N = 20
    # Number of discrete states for the ODE simulation (p_k's); must be at least > N
    Nde = 30
    # Number of independent trials (trajectories) to simulate
    K_trials = 10000
    # Reaction rates
    alpha = 1.0  # death rate for reaction: S -> I
    beta = 0.5  # birth rate for reaction: S -> 2S (must be smaller than alpha to ensure eventual extinction)

    # Create an array for the reaction rate constants for both reactions.
    # Reaction 1 (death) uses alpha; Reaction 2 (birth) uses beta.
    c = np.array([alpha, beta])

    # Define the change matrix (C) for the two reactions:
    # Each row represents the change in [S, I] caused by a reaction.
    # For reaction 1 (death): S decreases by 1, I increases by 1  -->  [-1, 1]
    # For reaction 2 (birth): S increases by 1, I remains the same       -->  [1, 0]
    C = np.array([[-1, 1], [1, 0]])

    # Initialize the state for each trial:
    # 's' will hold the number of individuals (S); start all trials with N individuals.
    # 'i' will hold the cumulative number of deaths (I); start with 0 for all trials.
    s = np.full(K_trials, N, dtype=int)
    i = np.zeros(K_trials, dtype=int)

    # We will store the history (trajectories) for S, I, and transition times (T).
    # Each will be stored as a list (with each element corresponding to a time step)
    S_traj = [s.copy()]  # List to store state S at each time step for all trials.
    I_traj = [i.copy()]  # List to store state I at each time step for all trials.
    T_traj = [np.zeros(K_trials)]  # List to store the simulation time for each trial.

    # Set up variables for the simulation loop.
    step = 0  # Counter for time steps.
    # Identify which trials are still "active" (i.e., have a population greater than 0).
    active_indices = np.where(s > 0)[0]
    Nt = len(active_indices)  # Number of trajectories not yet extinct
    # Set a maximum number of steps to prevent infinite loops in rare cases.
    max_steps = 5000

    # ============================================================
    # Begin the Gillespie simulation loop (stochastic simulation)
    # ============================================================
    while Nt > 0 and step < max_steps:
        step += 1  # increment the simulation step

        # --------------------------------------------------
        # Calculate reaction propensities for all trials
        # --------------------------------------------------
        # For each trial, the propensity for a reaction is the reaction rate constant multiplied by the number of individuals.
        # h_death: propensity for the death reaction S -> I.
        # h_birth: propensity for the birth reaction S -> 2S.
        h_death = c[0] * s
        h_birth = c[1] * s

        # Stack the two reaction propensities into one 2D array with shape (K_trials, 2)
        h = np.vstack((h_death, h_birth)).T

        # Compute the cumulative sum of the propensities along the reaction axis.
        # This cumulative sum will be used to select which reaction occurs.
        hc = np.cumsum(h, axis=1)

        # Total reaction rate for each trial is the sum of the two propensities.
        H = np.sum(h, axis=1)

        # --------------------------------------------------
        # Generate random numbers for time and reaction selection
        # --------------------------------------------------
        # For each active trajectory, generate two random numbers:
        #  - The first random number (r1) is used to sample the time to the next reaction.
        #  - The second random number (r2) is used to decide which reaction occurs.
        r = np.random.rand(Nt, 2)

        # Copy the last recorded times for all trials.
        # This array will be updated for active trials to record the time of the next reaction.
        T_current = T_traj[-1].copy()

        # --------------------------------------------------
        # Loop over each active (non-extinct) trajectory to update its state
        # --------------------------------------------------
        for idx in range(Nt):
            trial = active_indices[idx]

            # Update the time for the next reaction using an exponential waiting time:
            # time increment = -log(r1)/H, added to the current time.
            # (Note: Exponential waiting time is a standard method to sample reaction times.)
            if H[trial] > 0:
                T_current[trial] = T_current[trial] - np.log(r[idx, 0]) / H[trial]

            # Determine which reaction occurs by normalizing the cumulative propensities.
            # We then choose the first reaction for which r2 is below the normalized cumulative value.
            if H[trial] > 0:
                normalized_hc = hc[trial] / H[trial]
                # np.where returns the indices where the condition is True.
                reaction_index = np.where(r[idx, 1] <= normalized_hc)[0][0]
            else:
                reaction_index = (
                    None  # This case should not occur if the trial is active
                )

            # Update the state by applying the corresponding change from the reaction.
            if reaction_index is not None:
                s[trial] += C[reaction_index, 0]
                i[trial] += C[reaction_index, 1]

        # Append the updated states and times to the trajectory lists.
        S_traj.append(s.copy())
        I_traj.append(i.copy())
        T_traj.append(T_current.copy())

        # Update the list of active trials (only those with s > 0 are still simulated).
        active_indices = np.where(s > 0)[0]
        Nt = len(active_indices)

    # ============================================================
    # Convert trajectory lists to NumPy arrays for easier processing.
    # ============================================================
    # The arrays will have the shape (number_of_steps, K_trials)
    S_traj = np.array(S_traj)
    I_traj = np.array(I_traj)
    T_traj = np.array(T_traj)

    # ============================================================
    # Plot a sample trajectory (using the first trial) alongside a deterministic decay curve.
    # ============================================================
    sample_trial = 0
    t_sample = T_traj[:, sample_trial]  # Times for the sample trajectory.
    s_sample = S_traj[:, sample_trial]  # Population sizes for the sample trajectory.

    plt.figure(figsize=(8, 6))
    # Use a step plot to visualize the discrete jumps in the stochastic simulation.
    plt.step(
        t_sample, s_sample, where="post", label="Stochastic Trajectory", linewidth=2
    )

    # Plot the deterministic decay curve: N * exp(- (alpha-beta)*t)
    t_dense = np.linspace(0, t_sample[-1], 500)
    deterministic = N * np.exp(-(alpha - beta) * t_dense)
    plt.plot(t_dense, deterministic, "--", linewidth=2, label="Deterministic Decay")

    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Particle Number", fontsize=14)
    plt.title("Birth-Death Process: Sample Trajectory")
    plt.yticks(np.arange(0, N + 1, 4))
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the figure if the flag is set.
    if cmykflg:
        plt.savefig("b_d_traj.png", format="png")

    # ============================================================
    # Create a histogram for extinction times
    # ============================================================
    # Extinction time is defined as the time when each trial’s population reaches zero,
    # which is stored in the last column of the time trajectory array.
    extinction_times = T_traj[-1, :]

    # Create a histogram with 50 bins for the extinction times.
    NN, bin_edges = np.histogram(extinction_times, bins=50)
    # Compute bin centers from the edges to plot the histogram.
    TT = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    # Compute the average bin width (dt) to approximate the probability density function (PDF)
    dt = np.mean(np.diff(TT))
    # The approximate PDF is given by the normalized histogram counts.
    pdf_approx = NN / (K_trials * dt)

    plt.figure(figsize=(8, 6))
    plt.plot(TT, pdf_approx, "*", linewidth=4, label="Extinction Time PDF (Data)")
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.title("PDF for Extinction Times")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================================================
    # Solve the ODE system for extinction probabilities p_k (k = 0,...,Nde)
    # ============================================================
    # p0: initial probability vector (of length Nde+1)
    # In the MATLAB code, p0(N+1)=1 (using 1-based indexing); in Python (0-indexed), we set p0[N] = 1.
    p0 = np.zeros(Nde + 1)
    if N < (Nde + 1):
        p0[N] = 1
    else:
        print("Warning: initial state index exceeds number of ODE states.")

    # Set the time span for the ODE simulation.
    tstep = 0.1
    t_end = (
        extinction_times.max()
    )  # Use the maximum extinction time from the simulation.
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end + tstep, tstep)

    # Solve the stiff ODE system using the 'BDF' method (similar to MATLAB's ode23s).
    sol = solve_ivp(
        lambda t, p: de_rhs(t, p, Nde, alpha, beta),
        t_span,
        p0,
        method="BDF",
        t_eval=t_eval,
    )

    # ============================================================
    # Plot the extinction probability trajectories from the ODE solution.
    # ============================================================
    plt.figure(figsize=(8, 6))
    # In MATLAB, the loop is from j = 1 to Nde. Here, we loop over indices 1 to Nde (excluding state 0).
    for j in range(1, Nde + 1):
        plt.plot(sol.t, sol.y[j, :], label=f"p_{j}")
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Extinction Probability p_k", fontsize=14)
    plt.title("Extinction Probability Trajectories (ODE)")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)
    plt.show()

    # ============================================================
    # Compare the histogram for extinction times with the derivative from the ODE solution.
    # ============================================================
    plt.figure(figsize=(8, 6))
    plt.plot(TT, pdf_approx, "*", linewidth=4, label="Extinction Time PDF")
    # In MATLAB, the derivative dp0/dt is approximated as alpha*S(:,2). Here, S(:,2) corresponds to p_1 in our solution.
    plt.plot(
        sol.t, alpha * sol.y[1, :], "--", linewidth=2, label="alpha * p_1 (from ODE)"
    )
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Rate of Change of p_0", fontsize=14)
    plt.title("Comparison: Extinction Time PDF and dp_0/dt from ODE")
    plt.legend()
    plt.grid(True)
    plt.show()

    if cmykflg:
        plt.savefig("b_d_ext_hist.png", format="png")

    # ============================================================
    # Create a histogram for the number of births in each trial.
    # ============================================================
    # The number of births can be computed by subtracting the initial number of individuals (N) from I (which counts deaths plus births).
    births = I_traj[-1, :] - N
    # Determine the maximum number of births observed across trials.
    Nb = births.max()
    # Create an array to hold the count (histogram) for each possible number of births from 0 to Nb.
    HNb = np.zeros(Nb + 1, dtype=int)

    # Loop over each trial and update the histogram count for the corresponding number of births.
    for j in range(K_trials):
        # Since Python uses zero-based indexing, births[j] is already an appropriate index.
        HNb[births[j]] += 1

    # Create an array with the possible number of births for the x-axis.
    xnb = np.arange(0, Nb + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(xnb, HNb, "*")
    plt.xlabel("Number of Births", fontsize=14)
    plt.ylabel("Number of Trials", fontsize=14)
    plt.title("Histogram: Number of Births in Trials")
    plt.grid(True)
    plt.show()


# ------------------------------------------------------------
# Run the main simulation when the script is executed
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
