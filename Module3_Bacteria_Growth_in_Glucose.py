# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:29:43 2017

@author: Crissy
"""

import numpy as np
import matplotlib.pyplot as plt
import urllib
from scipy.integrate import odeint
from scipy.optimize import curve_fit

bacteria_data= urllib.request.urlopen("http://nemenmanlab.org/~ilya/images/d/d8/Bacteria.txt")
data_set=np.loadtxt(bacteria_data, delimiter=',')
data_set_sorted = data_set[data_set[:, 0].argsort()]

N=np.size(data_set)
Time=data_set_sorted[0:N:1,0]
Bacteria_Concentration=data_set_sorted[0:N:1,1]

plt.scatter(Time,Bacteria_Concentration)
plt.xlabel('Time[Hours]')
plt.ylabel('Bacteria[bacteria/ml]')
plt.title('Bacteria Concentration Over Time')
plt.show()
plt.figure()

plt.scatter(Time, n_data)
plt.xlabel('Time[Hours]')
plt.ylabel('Log of Bacteria [log bacteria/ml]')
plt.title('Log of Bacteria Concentration over Time')
plt.figure()

tf = 100
#initial glucose and bacteria concentrations
init = [0.2, 50]

# k = Monod constant = p*10^-100
k_guess = 0.007

t_data = np.transpose(data_set_sorted)[0]
n_data = np.log(np.transpose(data_set_sorted)[1])
n_original = np.transpose(data_set_sorted)[1]
t = np.linspace(0, tf, tf +1).flatten()

# mu = bacteria death rate
B = t_data[41:52]
B = np.vstack([np.ones(B.size), B]).T
y_intercept, slope = np.linalg.lstsq(B, n_data[41:52])[0]
mu_guess= -slope

# gmax = maximum bacteria growth rate
C = t_data[0:29]
C = C.reshape((C.size, 1))
z = n_data[0:29]
z = z.reshape(z.size, 1)
slope2 = float(np.linalg.lstsq(C, z)[0])
gmax_guess = slope2 + mu_guess

# a = CFU/mg*ml
#a_guess = 1 / (mu_guess * Bacteria_Concentration[36] / 0.2)
a_guess = (mu_guess * Bacteria_Concentration[41] / (0.2/38))

timearray = np.arange(0, 5, 1).flatten()
# Monod law = bacteria growth rate
def diff_eq(values, t, gmax, mu, k, a):
    p = values[0]
    n = values[1]
    for t in timearray:
        #lag time
        if t < 4.5:
            dn_dt = 0
            dp_dt = 0
        else:
            # n = bacteria concentration
            dn_dt = n * gmax * p / (p+k) - mu*n
            # p = glucose concentration
            dp_dt = - n * gmax * p / (p+k) * (1/a)
    return dp_dt, dn_dt
    
def odeint_diff_eq(t, gmax, mu, k, a):
    return odeint(diff_eq, init, timearray, args = (gmax, mu, k, a))[0:N:1,1]

def log_odeint_diff_eq(t, gmax, mu, k, a):
    return np.log(odeint(diff_eq, init, timearray, args = (gmax, mu, k, a)))[0:N:1,1]
    
gmax1, mu1, k1, a1 = curve_fit(log_odeint_diff_eq, t_data, n_data, p0 = (gmax_guess, mu_guess, k_guess, a_guess), bounds=([0,0,0,0], [1, 1, 1, 10000000000]))[0]

fieo = odeint(diff_eq, init, timearray, args = (gmax_guess, mu_guess, k_guess, a_guess))
y = log_odeint_diff_eq(t, gmax1, mu1, k1, a1)

#equations redefined to fit a larger time period
timearray2 = np.arange(4.5, 108.5, 1).flatten()
def diff_eq2(values, t, gmax, mu, k, a):
    p = values[0]
    n = values[1]
    for t in timearray2:
        #lag time
        if t < 4.5:
            dn_dt = 0
            dp_dt = 0
        else:
            # n = bacteria concentration
            dn_dt = n * gmax * p / (p+k) - mu*n
            # p = glucose concentration
            dp_dt = - n * gmax * p / (p+k) * (1/a)
    return dp_dt, dn_dt
    
def odeint_diff_eq2(t, gmax, mu, k, a):
    return odeint(diff_eq, init, timearray2, args = (gmax, mu, k, a))[0:N:1,1]

def log_odeint_diff_eq2(t, gmax, mu, k, a):
    return np.log(odeint(diff_eq, init, timearray2, args = (gmax, mu, k, a)))[0:N:1,1]

#graph odeint with parameter values from curve fit for bacteria vs. time graph
plt.plot(timearray2, log_odeint_diff_eq2(t, gmax1, mu1, k1, a1), label='log fit')
plt.xlim([0, 100])
plt.title('Log of Bacteria Concentration with Curvefit')
plt.xlabel('Time[Hours]')
plt.ylabel('Log of Bacteria Concentration')
plt.scatter(Time, n_data)
plt.show()
plt.figure()

plt.plot(t_data, n_original, 'o', label='Original data', markersize=4)
A = np.vstack([np.ones(len(t_data)), t_data, t_data**2, t_data**3, t_data**4, t_data**5]).T
a, b, c, d, e, f = np.linalg.lstsq(A, n_original)[0]
plt.plot(t, a+b*t+c*t**2+d*t**3+e*t**4+f*t**5, 'r', label='Fitted line')
plt.legend(loc=4)
plt.show()
plt.figure()

plt.plot(t_data, n_data, 'o', label='Log-Original data', markersize=4)
A = np.vstack([np.ones(len(t_data)), t_data, t_data**2, t_data**3, t_data**4, t_data**5]).T
a, b, c, d, e, f = np.linalg.lstsq(A, n_data)[0]
plt.plot(t, a+b*t+c*t**2+d*t**3+e*t**4+f*t**5, 'r', label='Fitted line')
plt.legend(loc=4)
plt.show()


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
