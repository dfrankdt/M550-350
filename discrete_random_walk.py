#!/usr/bin/env python3
"""
Discrete random walk

This script simulates a number of discrete random walks with probably of moving 2a

Figures produced
	- Figure 1: position after 100 time steps
	- Figure 2: mean squared displacement as a function of time step n compared to theory
	- Figure 3: comparison of histogram to Gaussian
	
Note that this code is based on discrete_random_walk.m
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Main Simulation Function
# =============================================================================
def discrete_random_walk():
	"""
	Perhaps add some comments
	"""
	# -- Simulation parameters
	alpha = 0.1		# probability
	N = 2000		# trials
	nt = 100		# time steps
	
	# -- Simulation tools
	X = np.zeros( (N, nt+1) )			# Positions
	tsteps = np.arange(nt+1)			# Time steps 
	c = np.array([alpha, 2*alpha, 1]) 	# vector to determine direction
	xm = np.array([1, -1, 0])			# vector to move R, L, to stay put
	
	# -- Pass through the nt time steps
	for j in tsteps-1:
		R = np.random.rand(N)
		mR = np.where(R<c[0], np.ones(np.size(R)), np.zeros(np.size(R)))
		mL = np.where((R>c[0])*(R<c[1]), np.ones(np.size(R)), np.zeros(np.size(R)))
		mZ = np.where(R>c[1], np.ones(np.size(R)), np.zeros(np.size(R)))
		X[:,j+1] = X[:,j] + mR*xm[0] + mL*xm[1] + mZ*xm[2]

	# -- Compute the mean-squared distance, also the variance when alpha = beta
	Xmd2 = np.mean(X**2, 0)
	
	# -- Collect the end positions to plot against a Gaussian
	XeDist, bins = np.histogram(X[:,-1], bins='auto', density=True)
	z = np.linspace(-15, 15, 2**9+1)
	p = 1/(np.sqrt(4*np.pi*alpha*nt)) * np.exp(-z**2/(4*alpha*nt))
	
	# -- Do some plotting
	plt.figure(1)
	plt.plot(X[:20,:].T)
	plt.xlabel('Time Step')
	plt.ylabel('Position')
	
	plt.figure(2)
	plt.plot(tsteps, Xmd2, label='Actual')
	plt.plot(tsteps, 2*alpha*tsteps, '--r', label='Theoretical')
	plt.xlabel('Time Step')
	plt.ylabel('Mean Squared Displacement')
	plt.legend(loc='upper left')
	
	plt.figure(3)
	plt.stairs(XeDist, bins)
	plt.plot(z, p)
	plt.show()
		
# =============================================================================
# Execute the simulation if the script is run directly.
# =============================================================================
if __name__ == "__main__":
    discrete_random_walk()


