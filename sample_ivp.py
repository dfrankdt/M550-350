import numpy as np

# Goal: Approximate the solution to u' = u(1-u) for a variety of initial conditions
#       Plot those approximate solutions as a function of t


from scipy import integrate
import matplotlib.pyplot as plt

# The following function is the right-hand side of the differential equation
#  t - scalar representing independent variable
#  y - state variable (possibly vector valued)

def dy(t, y):
	dy = y*(1-y)
	return dy

# Create t0 < t < tf as the t_span
# Create the initial conditions y(t0) = y0
tf = 5
t_span = [0, tf]

y0 = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]

# Do the solving
soln = integrate.solve_ivp(dy, t_span, y0, dense_output=True)

# Create a fine mesh t and interpolate the solution on that mesh
t = np.linspace(0, tf, 301)
u = soln.sol(t).T

# Do some plotting
plt.figure()
plt.plot(t, u)
#plt.plot(soln.t, soln.y.T)  # This would plot the non-interpolated solution
plt.xlabel('t')
plt.ylabel('u(t)')
plt.title('Solution of Logistic Equation')
plt.ylim((0, 2))
plt.grid()
plt.show()


