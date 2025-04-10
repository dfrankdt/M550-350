import numpy as np
import matplotlib.pyplot as plt
# Goal: Generate figure 1.2

u = np.linspace(-0.2, 1.2, 2**9+1)
t = range(-3,3)
one = np.ones(np.shape(t))
a = 10
alpha = 0.25

F = 1/(a*alpha*(1-alpha))*(-(1-alpha)*np.log(np.abs(u)) - alpha*np.log(np.abs(1-u)) + np.log(np.abs(u-alpha)))

plt.figure()
plt.plot(F, u)
plt.plot(t, one, '--', t, 0*one, '--', t, alpha*one, '--')
plt.axis([-3, 2, -0.2, 1.2])
plt.xlabel('t')
plt.ylabel('u(t)')
plt.show()
