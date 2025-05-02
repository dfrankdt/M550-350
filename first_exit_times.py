#!/usr/bin/env python3
"""
First exit time with a stochastic differential equation

This script simulates Brownian motion through a stochastic differential equation.

Figures produced:
 - Figure 1: Trajectories of ten such motions
 - Figure 2: A histogram of first exit times
 - Figure 3: Comparison of mean first exit time to predicted quadratic
 
Note that this code is based on first_exit_times.m
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Single exit time trajectory
# =============================================================================
def Xexit(x0, L, D):
	"""
	step forward with
	
		x_t+1 = x_t + dx_t	

	where dx_t = sqrt(2*D*dt) N(0, 1) for some small dt.
	
	Inputs: 
		x0 (float): Initial position
		L (float): Width of boundary
		D (float): Diffusion coefficient
		
	Outputs:
		t (ndarray): Times
		x (ndarray): Trajectory
	"""
	Nt = 6*2**8
	dt = 1/2**8
	
	t, x = 0, x0
	T, X = t, x

	rng = np.random.default_rng()
	while x < L:
		dx = np.sqrt(2*D*dt)*rng.normal(0, 1)
		x = np.where(x + dx < 0, -(x + dx), x + dx)
		t = t + dt
		## Note that np.append is probably not the best way to do this
		X = np.append(X, x)
		T = np.append(T, t)
	return T, X

# -- Simulation Parameters -	
D, L = 1, 1
Np = 1000
texit = np.zeros(Np)

# -- Plot some trajectories -
plt.figure(1)
for kp in range(4):
	t, x = Xexit(0, L, D)
	texit[kp] = np.max(t)
	plt.plot(t, x)
	plt.plot('t')
	plt.plot('x')

# -- Compute remaining trajectories without plotting -
for kp in range(4, Np):
	t, x = Xexit(0, L, D)
	texit[kp] = np.max(t)

# -- Do some statistics on the output
txdist, bins = np.histogram(texit, bins='auto', density=False)
plt.figure(2)
plt.stairs(txdist, bins)
plt.xlabel('Mean Exit Time from x=0')
plt.ylabel('Density')

# -- Collect exit time data for a number of initial points x0
texit = np.zeros(Np)
nx0 = 16
x0 = np.linspace(0.1, 0.9, nx0+1)
tm = np.zeros(nx0+1)
for kx0 in range(nx0+1):
	for kp in range(Np):
		t, x = Xexit(x0[kx0], L, D)
		texit[kp] = np.max(t)
	tm[kx0] = np.mean(texit)

# -- Plot the mean and compare it to the expected parabola
plt.figure(3)
plt.plot(x0, tm,'o', label='Stochastic Result')
plt.plot(x0, L**2/(2*D)*(1 - x0**2/L**2), '--r', label='Deterministic Result')
plt.ylim(0,1)
plt.xlabel('Initial Position')
plt.ylabel('Mean Exit Time')
plt.legend(loc='upper left')

plt.show()

# =============================================================================
# Execute the simulation if the script is run directly.
# =============================================================================
if __name__ == "__main__":
    first_exit_times()



