#!/usr/bin/env python3
"""
Exponential Decay via Gillespie Algorithm and ODE Simulation

This script simulates the exponential decay of particles through a stochastic process
(simulated with the Gillespie algorithm) and compares the results with the solution of an
ODE system describing the evolution of the probability distribution of particle numbers.

Reaction:
    S -> 0 with decay rate alpha

Figures produced:
  - Figure 1: A step plot of a single decay trajectory alongside a deterministic decay curve.
  - Figure 2: A histogram (as an approximate PDF) of extinction times compared to dp0/dt (ODE prediction)

Note that this code accomplishes what is referenced in exponential_delay_via_Gillespie.m
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

# =============================================================================
# Deterministic solution
# =============================================================================
def usoln(t, alpha, N):
	"""
	Compute the solution of the related deterministic system u(t) = N * exp(- alpha t)
	
	Inputs:
		t (ndarray): Time
		alpha (float): Decay rate
		N (int): Initial number of particles
	
	Returns:
		u (ndarray): Array of solution
	"""
	u = N*np.exp(-alpha * t)
	return u

# =============================================================================
# Stochastic solution
# =============================================================================
def ustochastic(alpha, N):
	"""
	Perform the Gillespie algorithm, computing the times t at which the transitions
	from k to k - 1 particles occur.
	
	Inputs:
		alpha (float): Decay rate
		N (int): Initial number of particles
		
	Returns:
		t (ndarray): times at which a transitions occur
		k (int array): number of particles at the time t
	"""
	R = np.random.rand(N)
	k = np.zeros(N+1)
	k[:-1] = np.arange(N, 0, -1)
	t = np.zeros(N+1)
	t[1:] = -1/alpha * np.cumsum( np.log(R) / k[:-1] )
	return t, k

def de_rhs(t, p, alpha):
	"""
	Compute the right hand side of the differential equation for p_k(t), given by
		pN' = - alpha * N * pN
		pk' = - alpha * k * pk + alpha * (k+1) * pk+1
		p0' = alpha * (1) * p1
	for k = 0, 1, 2, ..., N 
	
	Inputs:
		t (float): Time (unused since the DE is autonomous)
		p (ndarray): Array of probabilities for k = 0, 1, 2, ..., N
		alpha (float): Decay rate
		
	Returns:
		dp (ndarray): Array of time derivatives for each pk
	"""
	N = np.size(p) - 1
	k = np.arange(N+1)
	dp = np.zeros(N+1)
	dp[0] = alpha * (k[0] + 1) * p[1]
	dp[1:-1] = - alpha * k[1:-1] * p[1:-1] + alpha * ( k[1:-1] + 1 )*(p[2:]) 
	dp[-1] = -(k[-1] + 1) * alpha * p[-1]
	return dp

# =============================================================================
# Main Simulation Function
# =============================================================================
def stochastic_decay():
	"""
	Simulate exponential particle decay via the Gillespie algorithm and compare it
    with the ODE simulation of the probability distribution p_k(t).

    The reaction is: S -> 0 with rate alpha.

    Generates figures:
      - Figure 1: A step plot of a single decay trajectory alongside a deterministic decay curve.
	  - Figure 2: A histogram (as an approximate PDF) of extinction times compared to dp0/dt (ODE prediction)
	"""
	# -- Simulation Parameters -
	alpha = 1			# Decay rate
	N = 25				# Initial number of particles
	ktrials = 10000		# Number of independent trials

	# -- One Gillespie algorithm -
	[tzero, Ndecay] = ustochastic(alpha, N)
	t = np.linspace(0, np.ceil(max(tzero))+2, 2**8+1)

	# -- Independent Gillespie trials
	tlist = np.zeros( (ktrials, N+1) )
	for k in range(ktrials):
		[tlist[k,], Ndecay] = ustochastic(alpha, N)
	t0dist, bins = np.histogram(tlist[:,-1], 50, density=True)

	# -- Solution of the ODE system for pk
	tf = 12
	p0 = np.zeros(N+1)
	p0[-1] = 1

	soln = solve_ivp(de_rhs, [0, tf], p0, args=[alpha], dense_output=True)
	tt = np.linspace(0, tf, 2**8+1)
	p = soln.sol(tt).T
	dp0 = alpha*p[:,1]

	# -- Create plots
	plt.figure()
	plt.plot(t, usoln(t, alpha, N), '--r')
	plt.step(tzero, Ndecay, where='post')
	plt.xlabel(r"$\alpha t$")
	plt.ylabel('Particle Number')
	plt.title('Decay Trajectory: Stochastic vs. Deterministic')
	
	plt.figure()
	plt.plot(tt, dp0, '--r')
	plt.stairs(t0dist,bins)
	plt.xlabel(r"$\alpha t$")
	plt.ylabel(r"$dp_0/dt$")
	plt.title('Extinction: Data vs ODE Prediction')
	plt.show()
	
# =============================================================================
# Execute the simulation if the script is run directly.
# =============================================================================
if __name__ == "__main__":
    stochastic_decay()


