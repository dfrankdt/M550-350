import numpy as np

# Goal: Approximate the solution to x'' + ep x' + x = 0 for x(0) = 1, x'(0) = 0
#       Plot those approximate solutions as a function of t


from scipy import integrate
import matplotlib.pyplot as plt

# The following function is the right-hand side of the differential equation
#  t - scalar representing independent variable
#  y - state variable (possibly vector valued)
# ep - parameter we pass from the main code to the DE solver

def dy(t, y, ep):
	u, v = y
	du = v
	dv = -u - ep*v
	dy = [du, dv]
	return dy

# Create t0 < t < tf as the t_span
# Create the initial conditions y(t0) = y0
tf = 80
t_span = [0, tf]
y0 = [1, 0]	
ep = 0.1

# Do the solving
soln = integrate.solve_ivp(dy, t_span, y0, args=[ep], dense_output=True)

# Create a fine mesh t and interpolate the solution on that mesh
t = np.linspace(0, tf, 10001)
u = soln.sol(t).T
# Note: The array u contains x and x'
#  x  = u[:,0] 
#  x' = u[:,1]

# Do some plotting
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(t, u)
plt.xlabel('t')
plt.ylabel('u(t), v(t)')
plt.title('Solution Trajectories')
plt.grid()

plt.subplot(1,2,2)
plt.plot(u[:,0], u[:,1])
plt.xlabel('u(t)')
plt.ylabel('v(t)')
plt.title('Phase Plane')
plt.grid()

plt.show()


