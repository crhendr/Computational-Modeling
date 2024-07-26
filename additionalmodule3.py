# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:22:30 2017

@author: Crissy
"""
#for Michaelis Menten
#fits a line to data
#guesses values for constants
#finds the values for the constants by using curvefit
#does a log fit
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

x = np.arange(10)
y = np.array([-1, 0.2, 0.9, 2.1, 3.0, 3.9, 4.5, 4.9, 5.3, 6.1])
A = np.vstack([np.ones(len(x)), x, x**2, x**3, x**4, x**5]).T

a, b, c, d, e, f = np.linalg.lstsq(A, y)[0]

plt.plot(x, y, 'o', label='Original data', markersize=4)
x = np.arange(-10, 20, 1)
plt.plot(x, a+b*x+c*x**2+d*x**3+e*x**4+f*x**5, 'r', label='Fitted line')
plt.legend(loc=4)
plt.show()

init_conc = 10.0


def MichaelisMenten(x, t, V, K):
    return -V*x/(K+x)


def MMTrajectory(t, V, K):
    return odeint(MichaelisMenten, init_conc, t, args=(V, K)).flatten()


def logMMTrajectory(t, V, K):
    return np.log(odeint(MichaelisMenten, init_conc, t, args=(V, K)).flatten())

plt.figure()
t = np.linspace(0, 10, 30).flatten()
x = odeint(MichaelisMenten, init_conc, t, args=(3.3, 3.0)).flatten()
data = (1+0.2*np.random.randn(x.size))*x
plt.plot(t, x, label='simulated')
plt.plot(t, data, 'o', label='simulated noisy')
plt.legend()
plt.show()

first_range = 11
A = t[0:first_range]
A = A.reshape((A.size, 1))
y = data[0:first_range]-init_conc
y = y.reshape((y.size, 1))

a = float(np.linalg.lstsq(A, y)[0])
v_guess = -a


second_range = 15
A = t[second_range:]
A = np.vstack([np.ones(A.size), A]).T
y = np.log(data[second_range:])

a, b = np.linalg.lstsq(A, y)[0]
k_guess = -v_guess/b


v, k = curve_fit(MMTrajectory, t, data, p0=(v_guess, k_guess))[0]
vl, kl = curve_fit(logMMTrajectory, t, np.log(data), p0=(v_guess, k_guess))[0]

plt.figure()
plt.plot(t, data, 'o', label='data')
plt.plot(t, MMTrajectory(t, v, k), label='fit')
plt.plot(t, MMTrajectory(t, vl, kl), label='log fit')
plt.legend()
plt.show()

plt.figure()
plt.semilogy(t, data, 'o', label='data')
plt.semilogy(t, MMTrajectory(t, v, k), label='fit')
plt.semilogy(t, MMTrajectory(t, vl, kl), label='log fit')
plt.legend()
plt.show()

print('Estimated K=' + str(v_guess))
print('Estimated K=' + str(k_guess))
print('Fitted V=' + str(v))
print('Fitted K=' + str(k))
print('Log-fitted V=' + str(vl))
print('Log-fFitted K=' + str(kl))